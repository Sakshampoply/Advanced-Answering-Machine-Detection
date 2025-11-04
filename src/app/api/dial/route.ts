import { NextRequest, NextResponse } from "next/server";
import { PrismaClient } from "@prisma/client";
import { z } from "zod";
import twilio from "twilio";

const prisma = new PrismaClient();

// Input validation schema
const DialSchema = z.object({
  phone: z.string().regex(/^\+?[1-9]\d{1,14}$/, "Invalid phone number"),
  strategy: z.enum(["gemini-live", "twilio-amd", "jambonz", "hf-ml"]).default("gemini-live"),
  userId: z.number().optional(),
});

export async function POST(req: NextRequest) {
  try {
  const body = await req.json();
  const { phone, userId, strategy } = DialSchema.parse(body);

    console.log(`[Dial] Creating call to ${phone}`);
    console.log(`[Dial] TWILIO_ACCOUNT_SID: ${process.env.TWILIO_ACCOUNT_SID ? "SET" : "MISSING"}`);
    console.log(`[Dial] TWILIO_AUTH_TOKEN: ${process.env.TWILIO_AUTH_TOKEN ? "SET" : "MISSING"}`);
    console.log(`[Dial] TWILIO_PHONE_NUMBER: ${process.env.TWILIO_PHONE_NUMBER}`);
    console.log(`[Dial] NEXT_PUBLIC_API_URL: ${process.env.NEXT_PUBLIC_API_URL}`);

    // Initialize Twilio client
    const client = twilio(
      process.env.TWILIO_ACCOUNT_SID,
      process.env.TWILIO_AUTH_TOKEN
    );

    // Create a call with webhook for TwiML response
    // The TwiML will then enable Media Streams for real-time audio
  const webhookUrl = `${process.env.NEXT_PUBLIC_API_URL}/api/amd-webhook?strategy=${encodeURIComponent(strategy)}`;
  const recordingUrl = `${process.env.NEXT_PUBLIC_API_URL}/api/recording-status`;
  const amdStatusUrl = `${process.env.NEXT_PUBLIC_API_URL}/api/amd-status`;
  const callStatusUrl = `${process.env.NEXT_PUBLIC_API_URL}/api/call-status`;
    
    console.log(`[Dial] Webhook URL: ${webhookUrl}`);
    console.log(`[Dial] Recording URL: ${recordingUrl}`);
    
    // Base call options
    const callOpts: any = {
      url: webhookUrl,
      to: phone,
      from: process.env.TWILIO_PHONE_NUMBER || "",
      record: true,
      recordingStatusCallback: recordingUrl,
    };

    if (strategy === "twilio-amd") {
      // Enable Twilio Native AMD with async callbacks so TwiML can proceed while AMD runs
      callOpts.machineDetection = "Enable"; // Enable basic AMD (fast human/machine)
      // @ts-ignore - asyncAmd is supported on Calls create API for asynchronous AMD
      callOpts.asyncAmd = true; // ensure callbacks are posted to amdStatusCallback
      // Optional: cap detection time; ignored for some modes but safe to include
      callOpts.machineDetectionTimeout = 30; // seconds

      // AMD-specific callback carries AnsweringMachineDetectionStatus / AnsweredBy
      // @ts-ignore - Twilio SDK supports amdStatusCallback on Calls API
      callOpts.amdStatusCallback = amdStatusUrl;
      // @ts-ignore
      callOpts.amdStatusCallbackMethod = "POST";

      // Keep general call lifecycle separate
      callOpts.statusCallback = callStatusUrl;
      callOpts.statusCallbackEvent = ["initiated", "ringing", "answered", "completed"];
      callOpts.statusCallbackMethod = "POST";
    }

    const call = await client.calls.create(callOpts);

    // Log the call initiation in the database
    const callLogData: any = {
      phone,
      strategy: strategy === "twilio-amd" ? "Twilio AMD" : strategy === "hf-ml" ? "Hugging Face" : "Gemini Live",
      status: "Initiated",
      callSid: call.sid,
    };
    
    if (userId) {
      callLogData.userId = userId;
    }

    const callLog = await prisma.callLog.create({
      data: callLogData,
    });

    console.log(`Call initiated: ${call.sid} to ${phone}`);

    return NextResponse.json(
      {
        message: "Call initiated successfully",
        callSid: call.sid,
        callLogId: callLog.id,
      },
      { status: 200 }
    );
  } catch (error) {
    if (error instanceof z.ZodError) {
      return NextResponse.json(
        { error: "Invalid input", details: error.issues },
        { status: 400 }
      );
    }

    const errorMessage = error instanceof Error ? error.message : "Unknown error";
    console.error("Error initiating call:", error);
    
    // Provide helpful error messages
    let userMessage = "Failed to initiate call";
    if (errorMessage.includes("unverified")) {
      userMessage = "Phone number is not verified. Twilio trial accounts can only call verified numbers.";
    } else if (errorMessage.includes("invalid")) {
      userMessage = "Invalid phone number format";
    } else if (errorMessage.includes("authentication")) {
      userMessage = "Twilio authentication failed. Check your credentials.";
    }
    
    return NextResponse.json(
      { error: userMessage, details: errorMessage },
      { status: 400 }
    );
  } finally {
    await prisma.$disconnect();
  }
}
