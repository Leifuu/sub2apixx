package service

import (
	"encoding/json"
	"strings"

	"github.com/tidwall/gjson"
)

const (
	// 请求和续接状态各自最多 8 MiB；超限保持原有错误终态，不扩大重放范围。
	openAIContinuationByteLimit = 8 << 20
	openAIContinuationItemLimit = 128
)

type openAIStreamCheckpoint struct {
	items    []map[string]any
	done     []bool
	retained []json.RawMessage
	bytes    int
	blocked  string
	texts    map[openAIStreamPartKey]*strings.Builder
	textDone map[openAIStreamPartKey]bool
	partDone map[openAIStreamPartKey]bool
}

type openAIStreamPartKey struct {
	item, part int
	field      string
}

// 仅记录可安全回传模型的输出；工具一旦暴露给客户端，不在网关伪造工具结果。
func (s *openAIStreamCheckpoint) observe(payload []byte, kind string) {
	if s.blocked != "" || !strings.HasPrefix(kind, "response.") {
		return
	}
	if !gjson.GetBytes(payload, "output_index").Exists() {
		if strings.HasPrefix(kind, "response.output_") || strings.HasPrefix(kind, "response.content_part.") || strings.HasPrefix(kind, "response.reasoning_") {
			s.blocked = "missing_output_index"
		}
		return
	}
	index := gjson.GetBytes(payload, "output_index").Int()
	if index < 0 || index >= openAIContinuationItemLimit || len(s.retained)+int(index) >= openAIContinuationItemLimit {
		s.blocked = "item_limit"
		return
	}
	for len(s.items) <= int(index) {
		s.items = append(s.items, nil)
		s.done = append(s.done, false)
	}
	i := int(index)
	if kind == "response.output_item.added" || kind == "response.output_item.done" {
		raw := gjson.GetBytes(payload, "item").Raw
		var item map[string]any
		if json.Unmarshal([]byte(raw), &item) != nil || item == nil {
			s.blocked = "invalid_item"
			return
		}
		itemType, _ := item["type"].(string)
		if itemType != "message" && itemType != "reasoning" {
			s.blocked = "tool_or_unsupported_output"
			return
		}
		for _, part := range gjson.GetBytes(payload, "item.content").Array() {
			if part.Get("type").String() != "output_text" {
				s.blocked = "unsupported_content"
				return
			}
		}
		if kind == "response.output_item.added" && s.items[i] != nil {
			s.blocked = "duplicate_item_index"
			return
		}
		s.items[i] = item
		s.done[i] = kind == "response.output_item.done"
		for key := range s.texts {
			if key.item == i {
				delete(s.texts, key)
			}
		}
		s.bytes += len(raw)
	} else if s.items[i] == nil {
		s.blocked = "missing_item_start"
		return
	} else {
		item := s.items[i]
		field, partIndex := "content", gjson.GetBytes(payload, "content_index").Int()
		if strings.HasPrefix(kind, "response.reasoning_summary_") {
			field, partIndex = "summary", gjson.GetBytes(payload, "summary_index").Int()
		}
		if partIndex < 0 || partIndex >= openAIContinuationItemLimit {
			s.blocked = "part_limit"
			return
		}
		parts, _ := item[field].([]any)
		for len(parts) <= int(partIndex) {
			partType := "output_text"
			if field == "summary" {
				partType = "summary_text"
			}
			part := map[string]any{"type": partType, "text": ""}
			if field == "content" {
				part["annotations"] = []any{}
			}
			parts = append(parts, part)
		}
		part, ok := parts[partIndex].(map[string]any)
		if !ok {
			s.blocked = "invalid_part"
			return
		}
		key := openAIStreamPartKey{item: i, part: int(partIndex), field: field}
		switch kind {
		case "response.content_part.added", "response.content_part.done", "response.reasoning_summary_part.added", "response.reasoning_summary_part.done":
			var value map[string]any
			raw := gjson.GetBytes(payload, "part").Raw
			if json.Unmarshal([]byte(raw), &value) != nil || value == nil {
				s.blocked = "invalid_part"
				return
			}
			if value["type"] != "output_text" && value["type"] != "summary_text" {
				s.blocked = "unsupported_content"
				return
			}
			parts[partIndex] = value
			delete(s.texts, key)
			if strings.HasSuffix(kind, ".done") {
				if s.partDone == nil {
					s.partDone = make(map[openAIStreamPartKey]bool)
				}
				s.partDone[key] = true
			}
			s.bytes += len(raw)
		case "response.output_text.delta", "response.reasoning_summary_text.delta":
			delta := gjson.GetBytes(payload, "delta").String()
			if s.texts == nil {
				s.texts = make(map[openAIStreamPartKey]*strings.Builder)
			}
			builder := s.texts[key]
			if builder == nil {
				builder = &strings.Builder{}
				text, _ := part["text"].(string)
				builder.WriteString(text)
				s.texts[key] = builder
			}
			builder.WriteString(delta)
			s.bytes += len(delta)
		case "response.output_text.done", "response.reasoning_summary_text.done":
			text := gjson.GetBytes(payload, "text").String()
			part["text"] = text
			delete(s.texts, key)
			if s.textDone == nil {
				s.textDone = make(map[openAIStreamPartKey]bool)
			}
			s.textDone[key] = true
			s.bytes += len(text)
		case "response.refusal.delta", "response.refusal.done":
			// 拒答不可当作普通续写提示，保留原有失败处理。
			s.blocked = "refusal"
			return
		case "response.output_text.annotation.added":
			annotation := gjson.GetBytes(payload, "annotation")
			var value any
			if !annotation.IsObject() || json.Unmarshal([]byte(annotation.Raw), &value) != nil {
				s.blocked = "invalid_annotation"
				return
			}
			annotations, _ := part["annotations"].([]any)
			if len(annotations) >= openAIContinuationItemLimit {
				s.blocked = "annotation_limit"
				return
			}
			part["annotations"] = append(annotations, value)
			s.bytes += len(annotation.Raw)
		default:
			// 未实现的输出事件无法重建等价历史，保留原错误而不猜测其语义。
			s.blocked = "unsupported_output_event"
			return
		}
		item[field] = parts
	}
	if s.bytes > openAIContinuationByteLimit {
		s.blocked = "checkpoint_limit"
	}
}

// 失败时冻结已交付的前缀；未完成的文本项在旧流内明确收尾，续接项使用新索引。
func (s *openAIStreamCheckpoint) prepare() ([]json.RawMessage, []map[string]any, bool) {
	if s.blocked != "" {
		return nil, nil, false
	}
	for key, text := range s.texts {
		parts, _ := s.items[key.item][key.field].([]any)
		part, _ := parts[key.part].(map[string]any)
		part["text"] = text.String()
	}
	output := append([]json.RawMessage(nil), s.retained...)
	var closing []map[string]any
	for index, item := range s.items {
		if item == nil {
			return nil, nil, false
		}
		if !s.done[index] {
			itemType, _ := item["type"].(string)
			id, _ := item["id"].(string)
			field, partName, textName, indexKey := "content", "response.content_part.done", "response.output_text.done", "content_index"
			if itemType == "reasoning" {
				field, partName, textName, indexKey = "summary", "response.reasoning_summary_part.done", "response.reasoning_summary_text.done", "summary_index"
			}
			parts, _ := item[field].([]any)
			for partIndex, value := range parts {
				part, ok := value.(map[string]any)
				if !ok {
					return nil, nil, false
				}
				key := openAIStreamPartKey{item: index, part: partIndex, field: field}
				if !s.textDone[key] {
					closing = append(closing, map[string]any{"type": textName, "output_index": index, "item_id": id, indexKey: partIndex, "text": part["text"]})
				}
				if !s.partDone[key] {
					closing = append(closing, map[string]any{"type": partName, "output_index": index, "item_id": id, indexKey: partIndex, "part": part})
				}
			}
			// 这是失败尝试已经交付的前缀，不伪称模型完整完成了该消息。
			item["status"] = "incomplete"
			closing = append(closing, map[string]any{"type": "response.output_item.done", "output_index": index, "item": item})
		}
		raw, err := json.Marshal(item)
		if err != nil {
			return nil, nil, false
		}
		output = append(output, raw)
	}
	return output, closing, len(output) <= openAIContinuationItemLimit
}

func (s *openAIStreamCheckpoint) requestBody(original []byte, output []json.RawMessage) ([]byte, bool) {
	if gjson.GetBytes(original, "input.#").Int() > 4096 {
		return nil, false
	}
	var request map[string]json.RawMessage
	if json.Unmarshal(original, &request) != nil {
		return nil, false
	}
	var input []json.RawMessage
	if raw := request["input"]; len(raw) > 0 {
		if json.Unmarshal(raw, &input) != nil {
			var text string
			if json.Unmarshal(raw, &text) != nil {
				return nil, false
			}
			message, _ := json.Marshal(map[string]any{"role": "user", "content": text})
			input = append(input, message)
		}
	}
	for _, raw := range output {
		var item map[string]any
		if json.Unmarshal(raw, &item) != nil {
			return nil, false
		}
		if item["type"] == "reasoning" {
			// store=false 的推理必须带可验证的加密内容；不把摘要伪造成签名。
			if encrypted, _ := item["encrypted_content"].(string); encrypted == "" {
				continue
			}
		}
		delete(item, "status")
		encoded, err := json.Marshal(item)
		if err != nil {
			return nil, false
		}
		input = append(input, encoded)
	}
	encoded, err := json.Marshal(input)
	if err != nil {
		return nil, false
	}
	request["input"] = encoded
	if len(output) > 0 {
		// 恢复指令由网关产生，不提升上游文本的指令级别，也不伪造新的用户消息。
		var instructions string
		if raw := request["instructions"]; len(raw) > 0 && string(raw) != "null" && json.Unmarshal(raw, &instructions) != nil {
			return nil, false
		}
		instructions += "\n\nTransport recovery: the assistant output already in input was delivered before an interrupted generation. Continue the original task from that point without repeating the delivered text. Preserve all original task constraints."
		request["instructions"], _ = json.Marshal(instructions)
	}
	body, err := json.Marshal(request)
	return body, err == nil && len(body) <= openAIContinuationByteLimit
}

func (s *openAIStreamCheckpoint) commit(output []json.RawMessage) {
	s.retained = output
	s.items = nil
	s.done = nil
	s.texts = nil
	s.textDone = nil
	s.partDone = nil
}
