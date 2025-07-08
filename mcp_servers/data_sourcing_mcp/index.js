const express = require('express');
const axios = require('axios');

const app = express();
app.use(express.json());
const port = 3005;

// Example: Using NewsAPI. In a real scenario, this could be Bright Data, etc.
const NEWS_API_BASE_URL = 'https://newsapi.org/v2/everything';

app.post('/fetch-structured-data', async (req, res) => {
  const { source, query, language = 'en', sortBy = 'relevancy' } = req.body;
  const apiKey = process.env.NEWS_API_KEY;

  if (source !== 'market-news') {
    return res.status(400).json({ success: false, message: `Source '${source}' is not supported.` });
  }

  if (!query) {
    return res.status(400).json({ success: false, message: 'Missing query parameter' });
  }

  if (!apiKey) {
    return res.status(500).json({ success: false, message: 'NEWS_API_KEY environment variable not set' });
  }

  try {
    const response = await axios.get(NEWS_API_BASE_URL, {
      params: {
        q: query,
        language: language,
        sortBy: sortBy,
        apiKey: apiKey,
        pageSize: 10 // Limit the number of articles
      }
    });

    // Return the articles array which is structured data
    res.json({ success: true, data: response.data.articles });

  } catch (error) {
    console.error('Error fetching structured data:', error.response ? error.response.data : error.message);
    res.status(500).json({ success: false, message: error.message, details: error.response ? error.response.data : null });
  }
});

app.listen(port, () => {
  console.log(`Data Sourcing MCP Server listening on port ${port}`);
}); 