import { VercelRequest, VercelResponse } from "@vercel/node";

const DEEPSET_API_BASE = "https://api.cloud.deepset.ai/api/v1";

export default async function handler(req: VercelRequest, res: VercelResponse) {
  if (req.method !== "POST") {
    res.setHeader("Allow", "POST");
    return res.status(405).end("Method Not Allowed");
  }

  const { query, filter } = req.body;

  if (!query) {
    return res.status(400).json({ error: "Query is required" });
  }

  const { SEARCH_API_WORKSPACE, SEARCH_API_DEPLOYMENT, SEARCH_API_TOKEN } =
    process.env;

  if (!SEARCH_API_WORKSPACE || !SEARCH_API_DEPLOYMENT || !SEARCH_API_TOKEN) {
    console.error(
      "Search API environment variables are not configured on the server."
    );
    return res.status(500).json({ error: "Search service is not configured." });
  }

  try {
    const headers = {
      "Content-Type": "application/json",
      "X-Client-Source": "haystack-docs",
      Authorization: `Bearer ${SEARCH_API_TOKEN}`,
    };
    const deploymentUrl = `${DEEPSET_API_BASE}/workspaces/${SEARCH_API_WORKSPACE}/deployments/${SEARCH_API_DEPLOYMENT}`;

    // Deployment chat requires a search session
    const sessionResponse = await fetch(`${deploymentUrl}/search_sessions`, {
      method: "POST",
      headers,
    });
    if (!sessionResponse.ok) {
      console.error("Haystack API error:", await sessionResponse.text());
      return res
        .status(sessionResponse.status)
        .json({ error: `API error: ${sessionResponse.statusText}` });
    }
    const { search_session_id } = await sessionResponse.json();

    // Build the request body with optional filters
    const requestBody: any = {
      queries: [query],
      search_session_id,
    };

    // Add filters if provided (for future backend filtering support)
    if (filter && filter !== "all") {
      requestBody.debug = true;
      requestBody.filters = {
        operator: "AND",
        conditions: [
          {
            field: "meta.type",
            operator: "==",
            value: filter,
          },
        ],
      };
    }

    const apiResponse = await fetch(`${deploymentUrl}/chat`, {
      method: "POST",
      headers,
      body: JSON.stringify(requestBody),
    });

    if (!apiResponse.ok) {
      const errorData = await apiResponse.text();
      console.error("Haystack API error:", errorData);
      return res
        .status(apiResponse.status)
        .json({ error: `API error: ${apiResponse.statusText}` });
    }

    const data = await apiResponse.json();
    return res.status(200).json(data);
  } catch (error) {
    console.error("Internal server error:", error);
    return res.status(500).json({ error: "Failed to fetch search results." });
  }
}
