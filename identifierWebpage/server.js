const http = require('http');
const fs = require('fs');

const server = http.createServer((req, res) => {
  // Read the file from the file system
  fs.readFile('identifierWebpage/index.html', (err, data) => {
    if (err) {
      // If there is an error reading the file, send a 500 Internal Server Error response
      res.writeHead(500);
      res.end('Error reading file from file system');
      return;
    }

    // If the file was read successfully, send the contents of the file in the response
    res.writeHead(200);
    res.end(data);
  });
});

// Start the server listening on localhost:8080
server.listen(8080, 'localhost');