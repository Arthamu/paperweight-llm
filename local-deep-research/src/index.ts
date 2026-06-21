#!/usr/bin/env node
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequest,
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";

const SERPER_API_KEY = process.env.SERPER_API_KEY;
const LLAMA_ENDPOINT = process.env.LLAMA_ENDPOINT || "http://localhost:8080";
const LLAMA_MODEL = process.env.LLAMA_MODEL || "";
const CRAWL4AI_ENDPOINT = process.env.CRAWL4AI_ENDPOINT || "http://localhost:11235";
const MAX_CONTENT_LENGTH = 20000;

if (!SERPER_API_KEY) {
  console.error("Error: SERPER_API_KEY environment variable is required");
  process.exit(1);
}

interface SearchResult {
  title: string;
  url: string;
  description: string;
}

interface ScrapeResult {
  url: string;
  title: string;
  content: string;
  error?: string;
}

async function callLLM(systemPrompt: string, userPrompt: string): Promise<string> {
  const body: Record<string, unknown> = {
    messages: [
      { role: "system", content: systemPrompt },
      { role: "user", content: userPrompt },
    ],
    temperature: 0.3,
    max_tokens: 4096,
    stream: false,
  };
  if (LLAMA_MODEL) body.model = LLAMA_MODEL;

  const response = await fetch(`${LLAMA_ENDPOINT}/v1/chat/completions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    throw new Error(`LLM API error: ${response.status} ${response.statusText}`);
  }

  const data = await response.json() as { choices?: { message?: { content?: string } }[] };
  return data.choices?.[0]?.message?.content || "";
}

async function googleSearch(query: string, count: number = 5): Promise<SearchResult[]> {
  const response = await fetch("https://google.serper.dev/search", {
    method: "POST",
    headers: {
      "X-API-KEY": SERPER_API_KEY!,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ q: query, num: Math.min(count, 10), gl: "us", hl: "en" }),
  });

  if (!response.ok) {
    throw new Error(`Serper API error: ${response.status} ${response.statusText}`);
  }

  const data = await response.json() as { answerBox?: any; organic?: any[] };
  const organic = data.organic || [];

  if (data.answerBox) {
    return [
      {
        title: data.answerBox.title || "Featured Snippet",
        url: data.answerBox.link || "",
        description: data.answerBox.snippet || data.answerBox.answer || "",
      },
      ...organic.map((r: any) => ({
        title: r.title || "",
        url: r.link || "",
        description: r.snippet || "",
      })),
    ];
  }

  return organic.map((r: any) => ({
    title: r.title || "",
    url: r.link || "",
    description: r.snippet || "",
  }));
}

async function scrapePage(url: string): Promise<ScrapeResult> {
  try {
    const taskResp = await fetch(`${CRAWL4AI_ENDPOINT}/crawl/job`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        urls: [url],
        crawler_config: { scrape_delay: 0.5, max_pages: 1 },
      }),
    });
    if (!taskResp.ok) throw new Error(`crawl4ai submit error: ${taskResp.status}`);
    const { task_id } = await taskResp.json() as { task_id: string };

    let result: Record<string, unknown> | null = null;
    for (let i = 0; i < 30; i++) {
      const pollResp = await fetch(`${CRAWL4AI_ENDPOINT}/crawl/job/${task_id}`);
      if (!pollResp.ok) throw new Error(`crawl4ai poll error: ${pollResp.status}`);
      const pollData = await pollResp.json() as Record<string, unknown>;
      if (pollData.status === "completed") {
        result = pollData;
        break;
      }
      if (pollData.status === "failed") {
        throw new Error(`crawl4ai job failed: ${(pollData as any).error || "unknown"}`);
      }
      await new Promise(r => setTimeout(r, 2000));
    }
    if (!result) throw new Error("crawl4ai job timed out");

    const raw = result as any;
    const results: any[] | undefined =
      raw.results || raw.result?.results;
    if (!results || results.length === 0) throw new Error("crawl4ai returned no results");
    const page = results[0];
    if (!page.success) throw new Error(page.error_message || "crawl4ai scrape failed");

    const title = page.metadata?.title || url;
    let content = page.markdown?.raw_markdown || page.markdown?.fit_markdown || page.cleaned_html || page.fit_html || "";
    if (!content) throw new Error("crawl4ai returned no content");

    const stripped = content.replace(/<[^>]+>/g, " ").replace(/\s+/g, " ").trim();
    if (stripped.length > MAX_CONTENT_LENGTH) {
      const truncated = stripped.slice(0, MAX_CONTENT_LENGTH);
      const lastSentence = truncated.lastIndexOf(". ");
      content = (lastSentence > MAX_CONTENT_LENGTH * 0.8
        ? truncated.slice(0, lastSentence + 1)
        : truncated) + "... [truncated]";
    } else {
      content = stripped;
    }

    return { url, title, content };
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    return { url, title: "Error", content: "", error: message };
  }
}

async function generateSubQueries(query: string, depth: string): Promise<string[]> {
  const numQueries = depth === "quick" ? 2 : depth === "deep" ? 6 : 4;
  const prompt = `Given the research query: "${query}"

Generate ${numQueries} specific search sub-queries that would help comprehensively research this topic. Each sub-query should target a different aspect or angle.

Return ONLY a JSON array of strings, no other text. Example: ["sub-query 1", "sub-query 2"]`;

  try {
    const response = await callLLM(
      "You are a research planner. Always return valid JSON arrays.",
      prompt
    );
    const jsonMatch = response.match(/\[[\s\S]*\]/);
    if (jsonMatch) {
      return JSON.parse(jsonMatch[0]) as string[];
    }
    return JSON.parse(response) as string[];
  } catch {
    const depths: Record<string, string[]> = {
      quick: [query, `${query} key information`],
      moderate: [query, `${query} examples`, `${query} recent developments`, `${query} analysis`],
      deep: [
        query,
        `${query} overview`,
        `${query} examples and applications`,
        `${query} recent developments 2025 2026`,
        `${query} analysis and review`,
        `${query} comparison alternatives`,
      ],
    };
    return depths[depth] || depths.moderate;
  }
}

async function synthesizeReport(query: string, findings: string): Promise<string> {
  const prompt = `You are a research assistant producing a comprehensive research report.

Research Query: "${query}"

Below are findings from web searches and page scrapes. Synthesize them into a well-structured report.

FINDINGS:
${findings}

Produce a report with these sections:
1. **Executive Summary** - Brief overview of key findings
2. **Key Findings** - Detailed analysis organized by theme
3. **Supporting Evidence** - Specific facts, data, quotes with [Source: URL] notation
4. **Contrasting Views** - Different perspectives if applicable
5. **Conclusions** - Summary of what was learned

Always cite sources using [Source: URL] after each fact or claim. If findings are insufficient, state what is missing.`;

  try {
    return await callLLM(
      "You are a thorough research analyst producing detailed, well-cited reports.",
      prompt
    );
  } catch {
    return "";
  }
}

const server = new Server(
  { name: "local-deep-research", version: "1.0.0" },
  { capabilities: { tools: {} } }
);

server.setRequestHandler(ListToolsRequestSchema, async () => ({
  tools: [
    {
      name: "google_search",
      description:
        "Search Google for current information. Returns organic search results with titles, " +
        "URLs, and snippets. Use this for facts, news, documentation, or anything post-training cutoff. " +
        "Supports up to 10 results.",
      inputSchema: {
        type: "object",
        required: ["query"],
        properties: {
          query: {
            type: "string",
            description: "Search query",
          },
          count: {
            type: "number",
            description: "Number of results (1-10, default 5)",
            minimum: 1,
            maximum: 10,
          },
        },
      },
    },
    {
      name: "google_scrape",
      description:
        "Fetch and extract readable text content from a specific URL. Use this to read " +
        "webpages, documentation, articles, or any HTML page. Automatically cleans HTML, " +
        "removes scripts/styles, and truncates long content to fit context windows.",
      inputSchema: {
        type: "object",
        required: ["url"],
        properties: {
          url: {
            type: "string",
            description: "Full URL to scrape (e.g., https://example.com/docs)",
          },
        },
      },
    },
    {
      name: "deep_research",
      description: `Conduct deep, multi-step research on any topic. Orchestrates web searches, page scraping, and local LLM-powered synthesis into a structured report. Depth levels: quick (2 sub-queries), moderate (4), deep (6). Uses llama.cpp at LLAMA_ENDPOINT for query decomposition and report synthesis.`,
      inputSchema: {
        type: "object",
        required: ["query"],
        properties: {
          query: {
            type: "string",
            description: "The research question or topic to investigate",
          },
          depth: {
            type: "string",
            enum: ["quick", "moderate", "deep"],
            description: "Research depth (default: moderate)",
          },
          max_sources: {
            type: "number",
            description: "Maximum sources to scrape (default: 10, max: 25)",
            minimum: 1,
            maximum: 25,
          },
        },
      },
    },
  ],
}));

server.setRequestHandler(CallToolRequestSchema, async (request: CallToolRequest) => {
  try {
    const args = request.params.arguments as Record<string, unknown>;

    if (request.params.name === "google_search") {
      if (!args.query || typeof args.query !== "string") {
        throw new Error("Missing or invalid 'query'");
      }

      const count = typeof args.count === "number" ? args.count : 5;
      const results = await googleSearch(args.query, count);

      if (results.length === 0) {
        return { content: [{ type: "text", text: "No results found." }] };
      }

      const formatted = results
        .map((r, i) => `${i + 1}. **${r.title}**\n   ${r.url}\n   ${r.description}`)
        .join("\n\n");

      return {
        content: [{ type: "text", text: `Google results for "${args.query}":\n\n${formatted}` }],
      };
    }

    if (request.params.name === "google_scrape") {
      if (!args.url || typeof args.url !== "string") {
        throw new Error("Missing or invalid 'url'");
      }

      const result = await scrapePage(args.url);

      if (result.error) {
        return {
          content: [{ type: "text", text: `Failed to scrape ${result.url}: ${result.error}` }],
          isError: true,
        };
      }

      return {
        content: [{ type: "text", text: `**${result.title}**\n${result.url}\n\n${result.content}` }],
      };
    }

    if (request.params.name !== "deep_research") {
      throw new Error(`Unknown tool: ${request.params.name}`);
    }

    const query = args.query as string;
    const depth = (args.depth as string) || "moderate";
    const maxSources = Math.min((args.max_sources as number) || 10, 25);

    if (!query || typeof query !== "string") {
      throw new Error("Missing or invalid 'query'");
    }

    console.error("[Deep Research] Decomposing query into sub-queries...");
    const subQueries = await generateSubQueries(query, depth);

    console.error(`[Deep Research] Searching ${subQueries.length} sub-queries...`);
    const allResults: Array<SearchResult & { subQuery: string }> = [];
    for (const sq of subQueries) {
      try {
        const results = await googleSearch(sq, 5);
        for (const r of results) {
          allResults.push({ ...r, subQuery: sq });
        }
      } catch (e) {
        console.error(`[Deep Research] Search failed for: ${sq}`);
      }
    }

    if (allResults.length === 0) {
      return {
        content: [{ type: "text", text: "No search results found. Try a different query." }],
      };
    }

    const seen = new Set<string>();
    const uniqueResults = allResults.filter((r) => {
      if (seen.has(r.url)) return false;
      seen.add(r.url);
      return true;
    });

    const sourcesToScrape = uniqueResults.slice(0, maxSources);

    console.error(`[Deep Research] Scraping ${sourcesToScrape.length} sources...`);
    const scrapeResults = await Promise.all(
      sourcesToScrape.map(async (s) => {
        const scraped = await scrapePage(s.url);
        return { ...scraped, subQuery: s.subQuery, description: s.description };
      })
    );

    const successfulScrapes = scrapeResults.filter(
      (s): s is ScrapeResult & { subQuery: string; description: string } =>
        !s.error && s.content.length > 100
    );

    if (successfulScrapes.length === 0) {
      const searchOnly = uniqueResults
        .map((r, i) => `${i + 1}. **${r.title}**\n   ${r.url}\n   ${r.description}`)
        .join("\n\n");
      return {
        content: [{
          type: "text",
          text: `# Research: ${query}\n\nCould not scrape pages for detailed analysis. Here are search results:\n\n${searchOnly}`,
        }],
      };
    }

    console.error(`[Deep Research] Synthesizing ${successfulScrapes.length} sources with local LLM...`);
    const findingsText = successfulScrapes
      .map(
        (s) =>
          `--- Source: ${s.title} ---\nURL: ${s.url}\nSub-query: ${s.subQuery}\n\n${s.content.slice(0, 8000)}`
      )
      .join("\n\n");

    let report = await synthesizeReport(query, findingsText);

    if (!report) {
      const sourcesSection = successfulScrapes
        .map((s, i) => `${i + 1}. [${s.title}](${s.url}) - Related to: ${s.subQuery}`)
        .join("\n");
      report = `# Research Report: ${query}\n\n## Summary\n\nResearch completed with ${successfulScrapes.length} sources across ${subQueries.length} search dimensions.\n\n## Sources Consulted\n\n${sourcesSection}\n\n## Search Queries Used\n\n${subQueries.map((q, i) => `${i + 1}. ${q}`).join("\n")}`;
    }

    const sourceList = successfulScrapes
      .map((s, i) => `${i + 1}. [${s.title}](${s.url})`)
      .join("\n");

    const fullReport = `${report}\n\n---\n## References\n${sourceList}\n\n*Research depth: ${depth} | Sub-queries: ${subQueries.length} | Sources scraped: ${successfulScrapes.length}*`;

    return {
      content: [{ type: "text", text: fullReport }],
    };
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    console.error("Tool error:", message);
    return {
      content: [{ type: "text", text: JSON.stringify({ error: message }) }],
      isError: true,
    };
  }
});

async function runServer() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error("Local Deep Research MCP Server running on stdio");
  console.error(`LLM endpoint: ${LLAMA_ENDPOINT}`);
}

runServer().catch((error) => {
  console.error("Fatal error:", error);
  process.exit(1);
});
