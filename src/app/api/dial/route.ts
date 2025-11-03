import { NextRequest, NextResponse } from "next/server";
import { PrismaClient } from "@prisma/client";
import { z } from "zod";
import twilio from "twilio";

const prisma = new PrismaClient();

// Input validation schema
const DialSchema = z.object({
  phone: z.string().regex(/^\+?[1-9]\d{1,14}$/, "Invalid phone number"),
  userId: z.number().optional(),
});

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const { phone, userId } = DialSchema.parse(body);

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
    const webhookUrl = `${process.env.NEXT_PUBLIC_API_URL}/api/amd-webhook`;
    const recordingUrl = `${process.env.NEXT_PUBLIC_API_URL}/api/recording-status`;
    
    console.log(`[Dial] Webhook URL: ${webhookUrl}`);
    console.log(`[Dial] Recording URL: ${recordingUrl}`);
    
    const call = await client.calls.create({
      url: webhookUrl,
      to: phone,
      from: process.env.TWILIO_PHONE_NUMBER || "",
      record: true,
      recordingStatusCallback: recordingUrl,
    });

    // Log the call initiation in the database
    const callLogData: any = {
      phone,
      strategy: "Gemini Flash",
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
