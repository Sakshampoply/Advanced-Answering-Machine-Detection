import { NextRequest, NextResponse } from "next/server";
import twilio from "twilio";

export async function POST(req: NextRequest) {
  try {
    console.log(`[AMD Webhook] Request received`);
    console.log(`[AMD Webhook] Method: ${req.method}`);
    console.log(`[AMD Webhook] URL: ${req.url}`);
    const url = new URL(req.url);
    const strategy = url.searchParams.get("strategy") || "gemini-live";
    
    const formData = await req.formData();
    const callSid = formData.get("CallSid") as string | null;
    
    console.log(`[AMD Webhook] Call SID: ${callSid}`);
    console.log(`[AMD Webhook] Form data keys: ${Array.from(formData.keys()).join(", ")}`);

    // Create TwiML response (branch by strategy)
    const VoiceResponse = twilio.twiml.VoiceResponse;
    const twiml = new VoiceResponse();

  if (strategy === "twilio-amd") {
      // Baseline: rely on Twilio's native AMD; keep TwiML simple
      twiml.say({ voice: "alice" }, "Hello. Please hold.");
      twiml.pause({ length: 10 });
      // Twilio AMD runs before/alongside TwiML; results are posted to statusCallback
    } else {
      // Default: Media Streams for model-driven strategies (Gemini Live, Hugging Face)
      const mediaStreamPublicUrl = process.env.MEDIA_STREAM_PUBLIC_URL || process.env.NEXT_PUBLIC_API_URL;
      if (!mediaStreamPublicUrl) {
        console.error("MEDIA_STREAM_PUBLIC_URL or NEXT_PUBLIC_API_URL not set in .env");
        throw new Error("Missing Media Stream URL configuration");
      }

      // Convert to WebSocket URL - include strategy segment: /media-stream/{strategy}/{callSid}
      const strategySegment = strategy === "hf-ml" ? "hf" : "gemini";
      const mediaStreamUrl = `${mediaStreamPublicUrl.replace(/\/$/, "")}/media-stream/${strategySegment}/${callSid}`
        .replace(/^https/, "wss").replace(/^http/, "ws");

      console.log(`Media Stream URL for Twilio: ${mediaStreamUrl}`);

      const startVerb = twiml.start();
      startVerb.stream({ url: mediaStreamUrl });

      twiml.say({ voice: "alice" }, "Thank you for calling. One moment please.");
      twiml.gather({ numDigits: 1, timeout: 30 });
    }

    const twimlString = twiml.toString();
  console.log(`TwiML Response:\n${twimlString}`);
  console.log(`Responded with strategy: ${strategy} for call ${callSid}`);

    return new NextResponse(twiml.toString(), {
      headers: { "Content-Type": "application/xml" },
    });
  } catch (error) {
    console.error("Error in AMD webhook:", error);
    
    // Return a safe TwiML response on error
    const VoiceResponse = twilio.twiml.VoiceResponse;
    const twiml = new VoiceResponse();
    twiml.say("We experienced a technical issue. Goodbye.");
    twiml.hangup();

    return new NextResponse(twiml.toString(), {
      headers: { "Content-Type": "application/xml" },
      status: 500,
    });
  }
}
