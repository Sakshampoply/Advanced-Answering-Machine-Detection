import { NextRequest, NextResponse } from "next/server";
import twilio from "twilio";

export async function POST(req: NextRequest) {
  try {
    console.log(`[AMD Webhook] Request received`);
    console.log(`[AMD Webhook] Method: ${req.method}`);
    console.log(`[AMD Webhook] URL: ${req.url}`);
    
    const formData = await req.formData();
    const callSid = formData.get("CallSid") as string | null;
    
    console.log(`[AMD Webhook] Call SID: ${callSid}`);
    console.log(`[AMD Webhook] Form data keys: ${Array.from(formData.keys()).join(", ")}`);

    // Create TwiML response with Media Streams enabled
    const VoiceResponse = twilio.twiml.VoiceResponse;
    const twiml = new VoiceResponse();

    // Start Media Streams to capture audio in real-time
    // The Media Stream Server (port 3001) is exposed via ngrok for Twilio to reach it
    const mediaStreamPublicUrl = process.env.MEDIA_STREAM_PUBLIC_URL || process.env.NEXT_PUBLIC_API_URL;
    if (!mediaStreamPublicUrl) {
      console.error("MEDIA_STREAM_PUBLIC_URL or NEXT_PUBLIC_API_URL not set in .env");
      throw new Error("Missing Media Stream URL configuration");
    }
    
    // Convert to WebSocket URL - put callSid in path, not query string
    // (Twilio strips query strings on WebSocket upgrade)
    const mediaStreamUrl = `${mediaStreamPublicUrl.replace(/\/$/, "")}/media-stream/${callSid}`.replace(/^https/, "wss").replace(/^http/, "ws");
    
    console.log(`Media Stream URL for Twilio: ${mediaStreamUrl}`);
    
    // Use the Stream element through TwiML
    const startVerb = twiml.start();
    startVerb.stream({
      url: mediaStreamUrl,
    });

    // Say initial greeting while analyzing
    twiml.say({
      voice: "alice",
    }, "Thank you for calling. One moment please.");

    // Keep the call alive while analyzing
    twiml.gather({
      numDigits: 1,
      timeout: 30,
    });

    const twimlString = twiml.toString();
    console.log(`TwiML Response:\n${twimlString}`);
    console.log(`Started Media Stream for call ${callSid}`);

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
