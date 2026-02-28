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
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestInteractionsCreate(t *testing.T) {
	ctx := context.Background()

	handler := func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Errorf("expected POST request, got %s", r.Method)
		}
		if r.URL.Path != "/v1beta/interactions" {
			t.Errorf("expected path /v1beta/interactions, got %s", r.URL.Path)
		}

		resp := Interaction{
			ID:     "test-id",
			Status: "completed",
			Model:  "gemini-3-flash-preview",
			Outputs: []*InteractionContent{
				{
					Type: "text",
					Text: "Hello world!",
				},
			},
		}
		json.NewEncoder(w).Encode(resp)
	}

	server := httptest.NewServer(http.HandlerFunc(handler))
	defer server.Close()

	client, err := NewClient(ctx, &ClientConfig{
		APIKey: "test-api-key",
		HTTPOptions: HTTPOptions{
			BaseURL:    server.URL,
			APIVersion: "v1beta",
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	interaction := &Interaction{
		Model: "gemini-3-flash-preview",
		Input: "Hi",
	}

	resp, err := client.Interactions.Create(ctx, interaction, nil)
	if err != nil {
		t.Fatal(err)
	}

	if resp.ID != "test-id" {
		t.Errorf("expected ID test-id, got %s", resp.ID)
	}
	if len(resp.Outputs) != 1 || resp.Outputs[0].Text != "Hello world!" {
		t.Errorf("unexpected output: %+v", resp.Outputs)
	}
}

func TestInteractionsGet(t *testing.T) {
	ctx := context.Background()

	handler := func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			t.Errorf("expected GET request, got %s", r.Method)
		}
		if r.URL.Path != "/v1beta/interactions/test-id" {
			t.Errorf("expected path /v1beta/interactions/test-id, got %s", r.URL.Path)
		}

		resp := Interaction{
			ID:     "test-id",
			Status: "completed",
		}
		json.NewEncoder(w).Encode(resp)
	}

	server := httptest.NewServer(http.HandlerFunc(handler))
	defer server.Close()

	client, err := NewClient(ctx, &ClientConfig{
		APIKey: "test-api-key",
		HTTPOptions: HTTPOptions{
			BaseURL:    server.URL,
			APIVersion: "v1beta",
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	resp, err := client.Interactions.Get(ctx, "test-id", nil)
	if err != nil {
		t.Fatal(err)
	}

	if resp.ID != "test-id" {
		t.Errorf("expected ID test-id, got %s", resp.ID)
	}
}

func TestInteractionsCreateStream(t *testing.T) {
	ctx := context.Background()

	handler := func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Query().Get("alt") != "sse" {
			t.Errorf("expected alt=sse query param, got %s", r.URL.Query().Get("alt"))
		}

		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "keep-alive")

		resp1 := Interaction{
			ID:     "test-id",
			Status: "in_progress",
			Outputs: []*InteractionContent{
				{
					Type: "text",
					Text: "Part 1",
				},
			},
		}
		resp2 := Interaction{
			ID:     "test-id",
			Status: "completed",
			Outputs: []*InteractionContent{
				{
					Type: "text",
					Text: "Part 2",
				},
			},
		}

		data1, _ := json.Marshal(resp1)
		data2, _ := json.Marshal(resp2)

		fmt.Fprintf(w, "data: %s\n\n", data1)
		fmt.Fprintf(w, "data: %s\n\n", data2)
	}

	server := httptest.NewServer(http.HandlerFunc(handler))
	defer server.Close()

	client, err := NewClient(ctx, &ClientConfig{
		APIKey: "test-api-key",
		HTTPOptions: HTTPOptions{
			BaseURL:    server.URL,
			APIVersion: "v1beta",
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	interaction := &Interaction{
		Model: "gemini-3-flash-preview",
		Input: "Hi",
	}

	var texts []string
	for resp, err := range client.Interactions.CreateStream(ctx, interaction, nil) {
		if err != nil {
			t.Fatal(err)
		}
		if len(resp.Outputs) > 0 {
			texts = append(texts, resp.Outputs[0].Text)
		}
	}

	expected := []string{"Part 1", "Part 2"}
	if len(texts) != len(expected) {
		t.Fatalf("expected %d parts, got %d", len(expected), len(texts))
	}
	for i, v := range expected {
		if texts[i] != v {
			t.Errorf("expected part %d to be %s, got %s", i, v, texts[i])
		}
	}
}

func TestInteractionsDelete(t *testing.T) {
	ctx := context.Background()

	handler := func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodDelete {
			t.Errorf("expected DELETE request, got %s", r.Method)
		}
		if r.URL.Path != "/v1beta/interactions/test-id" {
			t.Errorf("expected path /v1beta/interactions/test-id, got %s", r.URL.Path)
		}
		w.WriteHeader(http.StatusNoContent)
	}

	server := httptest.NewServer(http.HandlerFunc(handler))
	defer server.Close()

	client, err := NewClient(ctx, &ClientConfig{
		APIKey: "test-api-key",
		HTTPOptions: HTTPOptions{
			BaseURL:    server.URL,
			APIVersion: "v1beta",
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	err = client.Interactions.Delete(ctx, "test-id", nil)
	if err != nil {
		t.Fatal(err)
	}
}

func TestInteractionsCancel(t *testing.T) {
	ctx := context.Background()

	handler := func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Errorf("expected POST request, got %s", r.Method)
		}
		if r.URL.Path != "/v1beta/interactions/test-id/cancel" {
			t.Errorf("expected path /v1beta/interactions/test-id/cancel, got %s", r.URL.Path)
		}

		resp := Interaction{
			ID:     "test-id",
			Status: "cancelled",
		}
		json.NewEncoder(w).Encode(resp)
	}

	server := httptest.NewServer(http.HandlerFunc(handler))
	defer server.Close()

	client, err := NewClient(ctx, &ClientConfig{
		APIKey: "test-api-key",
		HTTPOptions: HTTPOptions{
			BaseURL:    server.URL,
			APIVersion: "v1beta",
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	resp, err := client.Interactions.Cancel(ctx, "test-id", nil)
	if err != nil {
		t.Fatal(err)
	}

	if resp.Status != "cancelled" {
		t.Errorf("expected status cancelled, got %s", resp.Status)
	}
}
