import { createServer, request as httpRequest } from "http";
import { WebSocketServer, WebSocket as WSClient } from "ws";
import { PrismaClient } from "@prisma/client";
import type { IncomingMessage } from "http";
import type { Socket } from "net";

const prisma = new PrismaClient();
const PORT = process.env.WS_PORT || 3001;

// Map to track active streams
const activeStreams = new Map<string, any>();

// Create WebSocket server for Twilio Media Streams
const wss = new WebSocketServer({ noServer: true });

wss.on("connection", async (ws: any, req: IncomingMessage, callSid: string) => {
  console.log(`[Media Stream] New connection for call: ${callSid}`);

  let pythonWs: any = null;
  let buffer = Buffer.alloc(0);
  let analysisStarted = false;
  let detectionResult: any = null;
  let callStartTime: number | null = null;
  let callEndTime: number | null = null;

  try {
    // Connect to Python service WebSocket
    const pythonServiceUrl = process.env.PYTHON_SERVICE_URL || "http://localhost:8000";
    const pythonWsUrl = pythonServiceUrl.replace(/^http/, "ws") + `/ws/amd/${callSid}`;

    console.log(`[Media Stream] Connecting to Python service at: ${pythonWsUrl}`);

    // Use dynamic import for ws client
    const WebSocket = (await import("ws")).default;
    pythonWs = new WebSocket(pythonWsUrl);

    pythonWs.on("open", () => {
      console.log(`[Python WS] Connected for call ${callSid}`);
      analysisStarted = true;
    });

    pythonWs.on("message", (data: any) => {
      const message = JSON.parse(data.toString());
      console.log(`[Python Response] ${callSid}:`, message);

      if (message.type === "analysis_complete") {
        detectionResult = message;
        
        // Send decision back to Twilio via the main WebSocket
        // Twilio will use this to decide TwiML action
        ws.send(
          JSON.stringify({
            type: "control_frame",
            action: "stop",
            reason: `AMD ${message.result}: ${message.confidence}%`,
          })
        );

        // Update database with result
        updateCallResult(callSid, message);
      }
    });

    pythonWs.on("error", (error: any) => {
      console.error(`[Python WS] Error for call ${callSid}:`, error);
    });

    pythonWs.on("close", () => {
      console.log(`[Python WS] Closed for call ${callSid}`);
    });
  } catch (error) {
    console.error(`[Media Stream] Failed to connect to Python service:`, error);
  }

  ws.on("message", (data: Buffer) => {
    try {
      const message = JSON.parse(data.toString());

      if (message.event === "start") {
        console.log(`[Twilio Stream START] Call: ${message.start.callSid}`);
        callStartTime = Date.now();
      } else if (message.event === "media") {
        // Forward audio chunk to Python service
        const audioData = Buffer.from(message.media.payload, "base64");

        if (pythonWs && pythonWs.readyState === 1) {
          // WebSocket.OPEN
          pythonWs.send(
            JSON.stringify({
              type: "audio",
              data: audioData.toString("base64"),
              sample_rate: 8000,
              encoding: "PCMU",
            })
          );
        }
      } else if (message.event === "stop") {
        console.log(`[Twilio Stream STOP] Call: ${message.stop.callSid}`);
        callEndTime = Date.now();
        
        // Calculate duration in seconds
        const durationSeconds = callStartTime 
          ? Math.round((callEndTime - callStartTime) / 1000)
          : null;
        
        console.log(`[Call Duration] ${callSid}: ${durationSeconds}s`);
        
        // Update database with duration
        if (durationSeconds !== null) {
          updateCallDuration(callSid, durationSeconds);
        }
        
        // Clean up
        if (pythonWs) {
          pythonWs.send(JSON.stringify({ type: "end" }));
          pythonWs.close();
        }
        ws.close();
        activeStreams.delete(callSid);
      }
    } catch (error) {
      console.error(`[Media Stream] Error processing message for ${callSid}:`, error);
    }
  });

  ws.on("close", () => {
    console.log(`[Media Stream] Connection closed for call: ${callSid}`);
    if (pythonWs) {
      pythonWs.close();
    }
    activeStreams.delete(callSid);
  });

  ws.on("error", (error: any) => {
    console.error(`[Media Stream] WebSocket error for call ${callSid}:`, error);
  });

  activeStreams.set(callSid, ws);
});

async function updateCallResult(
  callSid: string,
  result: { result: string; confidence: number }
) {
  try {
    // Update by callSid (works at runtime even if Prisma types complain)
    await prisma.callLog.updateMany({
      where: { callSid } as any,
      data: {
        status: result.result.toLowerCase(),
        result: result.result,
        confidence: result.confidence,
      },
    });
    console.log(`[DB] Updated call ${callSid} with result:`, result);
  } catch (error) {
    console.error(`[DB] Failed to update call ${callSid}:`, error);
  }
}

async function updateCallDuration(callSid: string, durationSeconds: number) {
  try {
    await prisma.callLog.updateMany({
      where: { callSid } as any,
      data: {
        duration: durationSeconds,
      },
    });
    console.log(`[DB] Updated call ${callSid} with duration: ${durationSeconds}s`);
  } catch (error) {
    console.error(`[DB] Failed to update duration for call ${callSid}:`, error);
  }
}

// --- RE-IMPLEMENTING THE PROXY AND SERVER LOGIC ---

const server = createServer((req, res) => {
  // This function now acts as a proxy for the Next.js app.
  console.log(`[HTTP] ${req.method} ${req.url}`);

  const proxyReq = httpRequest({
    hostname: 'localhost',
    port: 3000, // Forward to Next.js app
    path: req.url,
    method: req.method,
    headers: req.headers,
  }, (proxyRes) => {
    res.writeHead(proxyRes.statusCode || 500, proxyRes.headers);
    proxyRes.pipe(res, { end: true });
  });

  proxyReq.on('error', (err) => {
    console.error(`[PROXY] Error: ${err.message}`);
    res.writeHead(502);
    res.end('Bad Gateway');
  });

  req.pipe(proxyReq, { end: true });
});

server.on('upgrade', (request: IncomingMessage, socket: Socket, head: Buffer) => {
  console.log(`[UPGRADE] Received upgrade request for URL: ${request.url}`);
  const pathname = request.url;

  if (pathname && pathname.startsWith('/media-stream/')) {
    wss.handleUpgrade(request, socket, head, (ws) => {
      const callSid = pathname.substring('/media-stream/'.length);
      console.log(`[UPGRADE] Success for call: ${callSid}`);
      wss.emit('connection', ws, request, callSid);
    });
  } else {
    console.log(`[UPGRADE] Invalid path. Destroying socket.`);
    socket.destroy();
  }
});

server.listen(PORT, () => {
  console.log(`[Server] Media Stream Server (with proxy) listening on port ${PORT}`);
});

process.on("SIGINT", () => {
  console.log("Shutting down Media Stream Server");
  server.close(() => {
    prisma.$disconnect();
    process.exit(0);
  });
});

export {};