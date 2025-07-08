const express = require('express');
const { google } = require('googleapis');
const { authorize } = require('./auth');
const stream = require('stream');

const app = express();
app.use(express.json({ limit: '50mb' }));
const port = 3001;

app.post('/upload', async (req, res) => {
  try {
    const { fileName, content } = req.body;
    if (!fileName || !content) {
      return res.status(400).send('Missing fileName or content');
    }

    const authClient = await authorize();
    const drive = google.drive({ version: 'v3', auth: authClient });

    const bufferStream = new stream.PassThrough();
    bufferStream.end(Buffer.from(content, 'utf-8'));

    const { data } = await drive.files.create({
      media: {
        mimeType: 'text/plain',
        body: bufferStream,
      },
      requestBody: {
        name: fileName,
        parents: [], // Can be configured to upload to a specific folder
      },
      fields: 'id',
    });

    console.log(`File uploaded successfully. File ID: ${data.id}`);
    res.json({ success: true, fileId: data.id });
  } catch (error) {
    console.error('Error uploading file:', error);
    res.status(500).json({ success: false, message: error.message });
  }
});

app.listen(port, () => {
  console.log(`Google Drive MCP Server listening on port ${port}`);
}); 