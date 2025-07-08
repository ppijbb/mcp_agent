const express = require('express');
const Figma = require('figma-api');

const app = express();
app.use(express.json());
const port = 3003;

app.get('/file-summary', async (req, res) => {
  const { fileId } = req.query;
  const figmaToken = process.env.FIGMA_TOKEN;

  if (!fileId) {
    return res.status(400).json({ success: false, message: 'Missing fileId query parameter' });
  }

  if (!figmaToken) {
    return res.status(500).json({ success: false, message: 'FIGMA_TOKEN environment variable not set' });
  }

  try {
    const api = new Figma.Api({
      personalAccessToken: figmaToken,
    });

    const file = await api.getFile(fileId, {
      depth: 1, // Only get top-level nodes to keep it fast
      plugin_data: 'none',
    });
    
    const summary = {
      name: file.name,
      lastModified: file.lastModified,
      version: file.version,
      pages: file.document.children.map(page => ({ id: page.id, name: page.name })),
      componentCount: Object.keys(file.components).length,
    };

    res.json({ success: true, data: summary });

  } catch (error) {
    console.error('Error fetching Figma file summary:', error);
    if (error.response && error.response.status === 404) {
      return res.status(404).json({ success: false, message: 'Figma file not found.' });
    }
    res.status(500).json({ success: false, message: 'An error occurred while fetching Figma data.' });
  }
});

app.listen(port, () => {
  console.log(`Figma Context MCP Server listening on port ${port}`);
}); 