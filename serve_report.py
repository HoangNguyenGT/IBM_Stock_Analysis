#!/usr/bin/env python
# Simple HTTP server to serve the HTML report

import http.server
import socketserver
import os
import sys

def serve_report(port=59169):
    """
    Serve the HTML report on the specified port.
    
    Args:
        port (int): Port number to serve on
    """
    # Change to the directory containing the report
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Check if report.html exists
    if not os.path.exists('report.html'):
        print("Error: report.html not found. Please generate the report first.")
        sys.exit(1)
    
    # Create a simple HTTP server
    handler = http.server.SimpleHTTPRequestHandler
    
    # Allow server to be accessed from any host
    with socketserver.TCPServer(("0.0.0.0", port), handler) as httpd:
        print(f"Serving IBM Stock Analysis Report at http://localhost:{port}/report.html")
        print("Press Ctrl+C to stop the server.")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")

if __name__ == "__main__":
    # Get port from command line argument if provided
    port = 59169
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print(f"Invalid port number: {sys.argv[1]}. Using default port {port}.")
    
    serve_report(port)