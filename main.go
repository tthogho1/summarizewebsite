package main

import (
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/api"
	"golang.org/x/net/html"
)

type SummaryRequest struct {
	URL string `json:"url"`
}

type SummaryResponse struct {
	Summary string `json:"summary"`
}

func wikiSummary(c *gin.Context) {
	var req SummaryRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(400, gin.H{"error": err.Error()})
		return
	}

	// Fetch text from the provided URL
	wikiText := fetchPageText(req.URL)

	// Summarize using Ollama LLM (phi3-mini)
	baseURL, err := url.Parse("http://localhost:11434")
	if err != nil {
		c.JSON(500, gin.H{"error": "Failed to parse Ollama URL: " + err.Error()})
		return
	}

	client := api.NewClient(baseURL, http.DefaultClient)
	stream := false
	var summary string
	if err := client.Generate(c.Request.Context(), &api.GenerateRequest{
		Model:  "phi3:mini",
		Prompt: fmt.Sprintf("Extract 5 key points from the following article:\n%s", wikiText),
		Stream: &stream,
		Options: map[string]any{
			"temperature": 0.1,
		},
	}, func(resp api.GenerateResponse) error {
		summary += resp.Response
		return nil
	}); err != nil {
		c.JSON(500, gin.H{"error": "Failed to generate summary: " + err.Error()})
		return
	}

	c.JSON(200, SummaryResponse{Summary: summary})
}

func fetchPageText(pageURL string) string {
	if pageURL == "" {
		return ""
	}
	if !strings.HasPrefix(pageURL, "http://") && !strings.HasPrefix(pageURL, "https://") {
		return ""
	}

	client := &http.Client{Timeout: 10 * time.Second}
	resp, err := client.Get(pageURL)
	if err != nil {
		return ""
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return ""
	}

	root, err := html.Parse(resp.Body)
	if err != nil {
		return ""
	}

	content := findByID(root, "mw-content-text")
	if content == nil {
		content = root
	}

	var paragraphs []string
	collectParagraphs(content, &paragraphs)
	if len(paragraphs) == 0 {
		return ""
	}

	return strings.Join(paragraphs, "\n\n")
}

func findByID(n *html.Node, id string) *html.Node {
	if n.Type == html.ElementNode {
		for _, a := range n.Attr {
			if a.Key == "id" && a.Val == id {
				return n
			}
		}
	}
	for c := n.FirstChild; c != nil; c = c.NextSibling {
		if found := findByID(c, id); found != nil {
			return found
		}
	}
	return nil
}

func collectParagraphs(n *html.Node, out *[]string) {
	if n.Type == html.ElementNode && n.Data == "p" {
		text := strings.TrimSpace(extractText(n))
		if text != "" {
			*out = append(*out, text)
		}
		return
	}
	for c := n.FirstChild; c != nil; c = c.NextSibling {
		collectParagraphs(c, out)
	}
}

func extractText(n *html.Node) string {
	if n.Type == html.ElementNode {
		if n.Data == "script" || n.Data == "style" || n.Data == "sup" {
			return ""
		}
	}
	if n.Type == html.TextNode {
		return strings.TrimSpace(n.Data)
	}
	var b strings.Builder
	for c := n.FirstChild; c != nil; c = c.NextSibling {
		part := extractText(c)
		if part == "" {
			continue
		}
		if b.Len() > 0 {
			b.WriteByte(' ')
		}
		b.WriteString(part)
	}
	return b.String()
}

func main() {
	r := gin.Default()
	r.POST("/summarize", wikiSummary)
	r.Run(":7860") // Gradio-compatible port
}
