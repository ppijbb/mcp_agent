const express = require('express');
const { getJson } = require("serpapi");

const app = express();
app.use(express.json());
const port = 3002;

app.post('/serp-analysis', async (req, res) => {
  const { query, engine = 'google' } = req.body;
  const apiKey = process.env.SERPAPI_KEY;

  if (!query) {
    return res.status(400).json({ success: false, message: 'Missing query parameter' });
  }

  if (!apiKey) {
    return res.status(500).json({ success: false, message: 'SERPAPI_KEY environment variable not set' });
  }

  try {
    const json = await getJson({
      q: query,
      engine: engine,
      api_key: apiKey
    });

    res.json({ success: true, data: json });

  } catch (error) {
    console.error('Error performing SERP analysis:', error);
    res.status(500).json({ success: false, message: error.message });
  }
});

app.listen(port, () => {
  console.log(`SEO MCP Server listening on port ${port}`);
}); 