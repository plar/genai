// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package genai

import (
	"context"
	"fmt"
	"iter"
	"net/http"
)

// Interactions provides access to the Interactions service.
type Interactions struct {
	apiClient *apiClient
}

// Interaction represents a generative AI interaction.
type Interaction struct {
	ID                    string                       `json:"id,omitempty"`
	Status                string                       `json:"status,omitempty"`
	Model                 string                       `json:"model,omitempty"`
	Agent                 string                       `json:"agent,omitempty"`
	Created               string                       `json:"created,omitempty"`
	Updated               string                       `json:"updated,omitempty"`
	Role                  string                       `json:"role,omitempty"`
	Outputs               []*InteractionContent        `json:"outputs,omitempty"`
	SystemInstruction     string                       `json:"systemInstruction,omitempty"`
	Tools                 []*InteractionTool           `json:"tools,omitempty"`
	Usage                 *InteractionUsage            `json:"usage,omitempty"`
	ResponseModalities    []ResponseModality           `json:"responseModalities,omitempty"`
	ResponseFormat        any                          `json:"responseFormat,omitempty"`
	ResponseMIMEType      string                       `json:"responseMimeType,omitempty"`
	PreviousInteractionID string                       `json:"previousInteractionId,omitempty"`
	Input                 any                          `json:"input,omitempty"` // Can be string, []*InteractionContent, []*InteractionTurn, or *InteractionContent
	GenerationConfig      *InteractionGenerationConfig `json:"generationConfig,omitempty"`
	AgentConfig           any                          `json:"agentConfig,omitempty"`
	Stream                bool                         `json:"stream,omitempty"`
	SDKHTTPResponse       *HTTPResponse                `json:"sdkHttpResponse,omitempty"`
}

// InteractionEvent represents an event in a streaming interaction.
type InteractionEvent struct {
	EventType   string              `json:"event_type"`
	Interaction *Interaction        `json:"interaction,omitempty"`
	Delta       *InteractionContent `json:"delta,omitempty"`
	Index       int                 `json:"index,omitempty"`
}

// ResponseModality represents the requested modality of the response.
type ResponseModality string

const (
	ResponseModalityText  ResponseModality = "text"
	ResponseModalityImage ResponseModality = "image"
	ResponseModalityAudio ResponseModality = "audio"
)

// InteractionContent is a polymorphic block used for both inputs and outputs.
type InteractionContent struct {
	Type        string                   `json:"type"`
	Text        string                   `json:"text,omitempty"`
	Annotations []*InteractionAnnotation `json:"annotations,omitempty"`
	Data        []byte                   `json:"data,omitempty"`
	URI         string                   `json:"uri,omitempty"`
	MIMEType    string                   `json:"mimeType,omitempty"`
	Resolution  MediaResolution          `json:"resolution,omitempty"`
	Signature   []byte                   `json:"signature,omitempty"`
	Summary     []*InteractionContent    `json:"summary,omitempty"`
	CallID      string                   `json:"callId,omitempty"`
	ID          string                   `json:"id,omitempty"`
	Name        string                   `json:"name,omitempty"`
	Arguments   any                      `json:"arguments,omitempty"`
	Result      any                      `json:"result,omitempty"`
	IsError     bool                     `json:"isError,omitempty"`
	ServerName  string                   `json:"serverName,omitempty"`
}

// InteractionAnnotation represents a source annotation for a text content.
type InteractionAnnotation struct {
	StartIndex int    `json:"startIndex,omitempty"`
	EndIndex   int    `json:"endIndex,omitempty"`
	Source     string `json:"source,omitempty"`
}

// InteractionTurn represents a single message in a conversation.
type InteractionTurn struct {
	Role    string `json:"role,omitempty"`
	Content any    `json:"content,omitempty"` // Can be string or []*InteractionContent
}

// InteractionTool represents a tool declaration.
type InteractionTool struct {
	Type                        string                   `json:"type"`
	Name                        string                   `json:"name,omitempty"`
	Description                 string                   `json:"description,omitempty"`
	Parameters                  any                      `json:"parameters,omitempty"`
	SearchTypes                 []string                 `json:"searchTypes,omitempty"`
	Environment                 string                   `json:"environment,omitempty"`
	ExcludedPredefinedFunctions []string                 `json:"excludedPredefinedFunctions,omitempty"`
	URL                         string                   `json:"url,omitempty"`
	Headers                     map[string]string        `json:"headers,omitempty"`
	AllowedTools                *InteractionAllowedTools `json:"allowedTools,omitempty"`
	FileSearchStoreNames        []string                 `json:"fileSearchStoreNames,omitempty"`
	TopK                        int                      `json:"topK,omitempty"`
	MetadataFilter              string                   `json:"metadataFilter,omitempty"`
}

// InteractionAllowedTools specifies which tools are allowed to be called.
type InteractionAllowedTools struct {
	Mode  string   `json:"mode,omitempty"`
	Tools []string `json:"tools,omitempty"`
}

// InteractionUsage statistics on the interaction request token usage.
type InteractionUsage struct {
	TotalInputTokens        int                         `json:"totalInputTokens,omitempty"`
	InputTokensByModality   []*InteractionModalityTokens `json:"inputTokensByModality,omitempty"`
	TotalCachedTokens       int                         `json:"totalCachedTokens,omitempty"`
	CachedTokensByModality  []*InteractionModalityTokens `json:"cachedTokensByModality,omitempty"`
	TotalOutputTokens       int                         `json:"totalOutputTokens,omitempty"`
	OutputTokensByModality  []*InteractionModalityTokens `json:"outputTokensByModality,omitempty"`
	TotalToolUseTokens      int                         `json:"totalToolUseTokens,omitempty"`
	ToolUseTokensByModality []*InteractionModalityTokens `json:"toolUseTokensByModality,omitempty"`
	TotalThoughtTokens      int                         `json:"totalThoughtTokens,omitempty"`
	TotalTokens             int                         `json:"totalTokens,omitempty"`
}

// InteractionModalityTokens token usage for a specific modality.
type InteractionModalityTokens struct {
	Modality ResponseModality `json:"modality,omitempty"`
	Tokens   int              `json:"tokens,omitempty"`
}

// InteractionGenerationConfig configuration parameters for the model interaction.
type InteractionGenerationConfig struct {
	Temperature       *float32                `json:"temperature,omitempty"`
	TopP              *float32                `json:"topP,omitempty"`
	Seed              *int32                  `json:"seed,omitempty"`
	StopSequences     []string                `json:"stopSequences,omitempty"`
	ThinkingLevel     string                  `json:"thinkingLevel,omitempty"`
	ThinkingSummaries string                  `json:"thinkingSummaries,omitempty"`
	MaxOutputTokens   int32                   `json:"maxOutputTokens,omitempty"`
	SpeechConfig      []*SpeechConfig         `json:"speechConfig,omitempty"`
	ImageConfig       *InteractionImageConfig `json:"imageConfig,omitempty"`
	ToolChoice        any                     `json:"toolChoice,omitempty"`
}

// InteractionImageConfig configuration for image generation.
type InteractionImageConfig struct {
	AspectRatio string `json:"aspectRatio,omitempty"`
	ImageSize   string `json:"imageSize,omitempty"`
}

// CreateInteractionConfig configuration for CreateInteraction.
type CreateInteractionConfig struct {
	HTTPOptions *HTTPOptions `json:"httpOptions,omitempty"`
}

// Create initiates a new generation.
func (i *Interactions) Create(ctx context.Context, interaction *Interaction, config *CreateInteractionConfig) (*Interaction, error) {
	var httpOptions *HTTPOptions
	if config == nil || config.HTTPOptions == nil {
		httpOptions = &HTTPOptions{}
	} else {
		httpOptions = config.HTTPOptions
	}

	path := "interactions"
	responseMap, err := sendRequest(ctx, i.apiClient, path, http.MethodPost, interaction, httpOptions)
	if err != nil {
		return nil, err
	}

	var response = new(Interaction)
	err = mapToStruct(responseMap, response)
	if err != nil {
		return nil, err
	}

	return response, nil
}

// CreateStream initiates a new generation and streams results.
func (i *Interactions) CreateStream(ctx context.Context, interaction *Interaction, config *CreateInteractionConfig) iter.Seq2[*InteractionEvent, error] {
	var httpOptions *HTTPOptions
	if config == nil || config.HTTPOptions == nil {
		httpOptions = &HTTPOptions{}
	} else {
		httpOptions = config.HTTPOptions
	}

	interaction.Stream = true
	path := "interactions?alt=sse"
	var rs responseStream[InteractionEvent]

	err := sendStreamRequest(ctx, i.apiClient, path, http.MethodPost, interaction, httpOptions, &rs)
	if err != nil {
		return yieldErrorAndEndIterator[InteractionEvent](err)
	}

	return iterateResponseStream(&rs, func(responseMap map[string]any) (*InteractionEvent, error) {
		var response = new(InteractionEvent)
		err = mapToStruct(responseMap, response)
		if err != nil {
			return nil, err
		}
		return response, nil
	})
}

// Get fetches the full state of an interaction.
func (i *Interactions) Get(ctx context.Context, id string, config *GetInteractionConfig) (*Interaction, error) {
	var httpOptions *HTTPOptions
	if config == nil || config.HTTPOptions == nil {
		httpOptions = &HTTPOptions{}
	} else {
		httpOptions = config.HTTPOptions
	}

	path := fmt.Sprintf("interactions/%s", id)
	responseMap, err := sendRequest(ctx, i.apiClient, path, http.MethodGet, nil, httpOptions)
	if err != nil {
		return nil, err
	}

	var response = new(Interaction)
	err = mapToStruct(responseMap, response)
	if err != nil {
		return nil, err
	}

	return response, nil
}

// GetStream streams a previously created background interaction or resumes a stream.
func (i *Interactions) GetStream(ctx context.Context, id string, config *GetInteractionConfig) iter.Seq2[*InteractionEvent, error] {
	var httpOptions *HTTPOptions
	if config == nil || config.HTTPOptions == nil {
		httpOptions = &HTTPOptions{}
	} else {
		httpOptions = config.HTTPOptions
	}

	path := fmt.Sprintf("interactions/%s?alt=sse", id)
	if config != nil && config.LastEventID != "" {
		path = fmt.Sprintf("%s&last_event_id=%s", path, config.LastEventID)
	}

	var rs responseStream[InteractionEvent]
	err := sendStreamRequest(ctx, i.apiClient, path, http.MethodGet, nil, httpOptions, &rs)
	if err != nil {
		return yieldErrorAndEndIterator[InteractionEvent](err)
	}

	return iterateResponseStream(&rs, func(responseMap map[string]any) (*InteractionEvent, error) {
		var response = new(InteractionEvent)
		err = mapToStruct(responseMap, response)
		if err != nil {
			return nil, err
		}
		return response, nil
	})
}

// Delete removes the interaction resource from the server.
func (i *Interactions) Delete(ctx context.Context, id string, config *DeleteInteractionConfig) error {
	var httpOptions *HTTPOptions
	if config == nil || config.HTTPOptions == nil {
		httpOptions = &HTTPOptions{}
	} else {
		httpOptions = config.HTTPOptions
	}

	path := fmt.Sprintf("interactions/%s", id)
	_, err := sendRequest(ctx, i.apiClient, path, http.MethodDelete, nil, httpOptions)
	return err
}

// Cancel stops a running background interaction.
func (i *Interactions) Cancel(ctx context.Context, id string, config *CancelInteractionConfig) (*Interaction, error) {
	var httpOptions *HTTPOptions
	if config == nil || config.HTTPOptions == nil {
		httpOptions = &HTTPOptions{}
	} else {
		httpOptions = config.HTTPOptions
	}

	path := fmt.Sprintf("interactions/%s/cancel", id)
	responseMap, err := sendRequest(ctx, i.apiClient, path, http.MethodPost, nil, httpOptions)
	if err != nil {
		return nil, err
	}

	var response = new(Interaction)
	err = mapToStruct(responseMap, response)
	if err != nil {
		return nil, err
	}

	return response, nil
}

// GetInteractionConfig configuration for GetInteraction.
type GetInteractionConfig struct {
	HTTPOptions *HTTPOptions `json:"httpOptions,omitempty"`
	LastEventID string       `json:"lastEventId,omitempty"`
}

// DeleteInteractionConfig configuration for DeleteInteraction.
type DeleteInteractionConfig struct {
	HTTPOptions *HTTPOptions `json:"httpOptions,omitempty"`
}

// CancelInteractionConfig configuration for CancelInteraction.
type CancelInteractionConfig struct {
	HTTPOptions *HTTPOptions `json:"httpOptions,omitempty"`
}

// interactionToMap converts an Interaction struct to a map for the API request.
func interactionToMap(i *Interaction) map[string]any {
	m := make(map[string]any)
	deepMarshal(i, &m)
	return m
}
