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

wss.on("connection", async (ws: any, req: IncomingMessage, callSid: string, strategy: "gemini" | "hf" = "gemini") => {
  // Back-compat guard: older upgrade logic may pass callSid as "hf/{sid}" or "gemini/{sid}"
  if (callSid.includes("/")) {
    const [maybeStrat, sid] = callSid.split("/");
    if ((maybeStrat === "hf" || maybeStrat === "gemini") && sid) {
      strategy = maybeStrat as any;
      callSid = sid;
    }
  }
  console.log(`[Media Stream] New connection for call: ${callSid} (strategy=${strategy})`);

  let pythonWs: any = null;
  let buffer = Buffer.alloc(0);
  let analysisStarted = false;
  let detectionResult: any = null;
  let callStartTime: number | null = null;
  let callEndTime: number | null = null;

  try {
    // Connect to Python service WebSocket
  const pythonServiceUrl = process.env.PYTHON_SERVICE_URL || "http://localhost:8000";
  const pathSeg = strategy === "hf" ? `/ws/hf/${callSid}` : `/ws/amd/${callSid}`;
  const pythonWsUrl = pythonServiceUrl.replace(/^http/, "ws") + pathSeg;

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
        // Human-readable percentage in reason (Twilio ignores it functionally but helpful for logs)
        const pct = typeof message.confidence === 'number' ? (message.confidence * 100).toFixed(0) : '—';
        ws.send(
          JSON.stringify({
            type: "control_frame",
            action: "stop",
            reason: `AMD ${message.result}: ${pct}%`,
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

  ws.on("message", async (data: Buffer) => {
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
        
        // Politely signal end to analyzer and wait briefly for a final decision
        const waitForDecision = async () => {
          try {
            if (pythonWs && pythonWs.readyState === 1) {
              pythonWs.send(JSON.stringify({ type: "end" }));
            }
            const started = Date.now();
            // Allow more time for HF strategy to finish local/remote ASR
            const maxWaitMs = strategy === 'hf' ? 12000 : 2500;
            while (!detectionResult && Date.now() - started < maxWaitMs) {
              await new Promise((r) => setTimeout(r, 50));
            }
          } catch {}
        };
        await waitForDecision();

        // Clean up
        try { if (pythonWs) pythonWs.close(); } catch {}
        try { ws.close(); } catch {}
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
    // Support both /media-stream/{callSid} (backward compat) and /media-stream/{strategy}/{callSid}
    const parts = pathname.split('/').filter(Boolean); // ["media-stream", maybeStrategy, callSid]
    let strategy: "gemini" | "hf" = "gemini";
    let callSid = "";

    if (parts.length === 2) {
      // /media-stream/{callSid}
      callSid = parts[1];
    } else if (parts.length >= 3) {
      // /media-stream/{strategy}/{callSid}
      strategy = parts[1] === 'hf' ? 'hf' : 'gemini';
      callSid = parts[2];
    }

    wss.handleUpgrade(request, socket, head, (ws) => {
      console.log(`[UPGRADE] Success for call: ${callSid}, strategy=${strategy}`);
      wss.emit('connection', ws, request, callSid, strategy);
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