// Files under `api/` become HTTP endpoints automatically — this is `/api/mcp`.
import { VercelRequest, VercelResponse } from "@vercel/node";

export default async function handler(req: VercelRequest, res: VercelResponse) {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
  res.setHeader(
    "Access-Control-Allow-Headers",
    "Content-Type, Accept, Mcp-Session-Id, Mcp-Protocol-Version"
  );
  res.setHeader("Access-Control-Expose-Headers", "Mcp-Session-Id");

  if (req.method === "OPTIONS") {
    return res.status(204).end();
  }

  if (req.method !== "POST") {
    res.setHeader("Allow", "POST, OPTIONS");
    return res.status(405).end("Method Not Allowed");
  }

  // Get environment variables
  const { MCP_WORKSPACE_ID, SEARCH_API_TOKEN } = process.env;

  if (!MCP_WORKSPACE_ID || !SEARCH_API_TOKEN) {
    return res.status(500).json({ error: "MCP service is not configured." });
  }

  try {
    // Forward the JSON-RPC body unchanged with the API key injected, so we
    // don't need to know any MCP methods — new upstream tools just work.
    const apiResponse = await fetch(
      `https://api.cloud.deepset.ai/api/v2/workspaces/${MCP_WORKSPACE_ID}/mcp`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept:
            (req.headers.accept as string) ||
            "application/json, text/event-stream",
          "X-Client-Source": "haystack-docs",
          Authorization: `Bearer ${SEARCH_API_TOKEN}`,
          // Forward MCP session id so upstream can correlate requests.
          ...(req.headers["mcp-session-id"] && {
            "Mcp-Session-Id": req.headers["mcp-session-id"] as string,
          }),
          // Forward the protocol version the client negotiated.
          ...(req.headers["mcp-protocol-version"] && {
            "Mcp-Protocol-Version": req.headers[
              "mcp-protocol-version"
            ] as string,
          }),
        },
        body: JSON.stringify(req.body),
      }
    );

    // Pass the response through as-is (status, content-type, raw body).
    const text = await apiResponse.text();
    res.status(apiResponse.status);
    const contentType = apiResponse.headers.get("content-type");
    if (contentType) res.setHeader("Content-Type", contentType);
    // Surface the session id back to the client (browsers can read it because
    // it's in Access-Control-Expose-Headers above).
    const sessionId = apiResponse.headers.get("mcp-session-id");
    if (sessionId) res.setHeader("Mcp-Session-Id", sessionId);
    return res.send(text);
  } catch (error) {
    console.error("MCP proxy error:", error);
    return res.status(502).json({ error: "Failed to reach MCP upstream." });
  }
}
