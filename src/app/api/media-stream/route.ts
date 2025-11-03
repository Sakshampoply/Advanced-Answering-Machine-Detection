import { NextRequest } from "next/server";

// This route handles WebSocket upgrade requests from Twilio
// It needs to forward the upgrade to the standalone media-stream-server on port 3001

export async function GET(req: NextRequest) {
  try {
    const callSid = req.nextUrl.searchParams.get("callSid");

    if (!callSid) {
      return new Response("Missing callSid", { status: 400 });
    }

    console.log(`[/api/media-stream] Received request for call: ${callSid}`);

    // Twilio will send WebSocket upgrade headers
    const upgrade = req.headers.get("upgrade");
    const connection = req.headers.get("connection");

    if (upgrade?.toLowerCase() === "websocket" && connection?.toLowerCase().includes("upgrade")) {
      console.log(`[/api/media-stream] WebSocket upgrade detected for ${callSid}`);
      
      // Forward to the standalone media stream server on port 3001
      // The standalone server will handle the actual WebSocket connection
      const targetUrl = `http://localhost:3001/media-stream?callSid=${callSid}`;
      
      console.log(`[/api/media-stream] Forwarding to: ${targetUrl}`);

      try {
        // Create a proxy request to the standalone server
        const proxyReq = await fetch(targetUrl, {
          method: "GET",
          headers: {
            "Connection": "Upgrade",
            "Upgrade": "websocket",
            "Sec-WebSocket-Key": req.headers.get("sec-websocket-key") || "",
            "Sec-WebSocket-Version": req.headers.get("sec-websocket-version") || "13",
          },
        } as any);

        return proxyReq;
      } catch (proxyError) {
        console.error(`[/api/media-stream] Proxy error:`, proxyError);
        return new Response("Failed to connect to media stream server", { status: 503 });
      }
    } else {
      return new Response(
        "Expected WebSocket upgrade request with Connection: Upgrade header",
        { status: 400 }
      );
    }
  } catch (error) {
    console.error("[/api/media-stream] Error:", error);
    return new Response("Internal Server Error", { status: 500 });
  }
}

