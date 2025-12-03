import { VercelRequest, VercelResponse } from "@vercel/node";

export default async function handler(req: VercelRequest, res: VercelResponse) {
  if (req.method !== "POST") {
    res.setHeader("Allow", "POST");
    return res.status(405).end("Method Not Allowed");
  }

  const { query, filter } = req.body;

  if (!query) {
    return res.status(400).json({ error: "Query is required" });
  }

  const { SEARCH_API_WORKSPACE, SEARCH_API_PIPELINE, SEARCH_API_TOKEN } =
    process.env;

  if (!SEARCH_API_WORKSPACE || !SEARCH_API_PIPELINE || !SEARCH_API_TOKEN) {
    console.error(
      "Search API environment variables are not configured on the server."
    );
    return res.status(500).json({ error: "Search service is not configured." });
  }

  try {
    // Build the request body with optional filters
    const requestBody: any = {
      queries: [query],
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

    const apiResponse = await fetch(
      `https://api.cloud.deepset.ai/api/v1/workspaces/${SEARCH_API_WORKSPACE}/pipelines/${SEARCH_API_PIPELINE}/search`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-Client-Source": "haystack-docs",
          Authorization: `Bearer ${SEARCH_API_TOKEN}`,
        },
        body: JSON.stringify(requestBody),
      }
    );

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
