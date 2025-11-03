import { NextRequest, NextResponse } from "next/server";

export async function GET(req: NextRequest) {
  return NextResponse.json({
    twilio_account_sid: process.env.TWILIO_ACCOUNT_SID ? "SET" : "MISSING",
    twilio_auth_token: process.env.TWILIO_AUTH_TOKEN ? "SET" : "MISSING",
    twilio_phone_number: process.env.TWILIO_PHONE_NUMBER || "MISSING",
    next_public_api_url: process.env.NEXT_PUBLIC_API_URL || "MISSING",
    python_service_url: process.env.PYTHON_SERVICE_URL || "MISSING",
    media_stream_public_url: process.env.MEDIA_STREAM_PUBLIC_URL || "MISSING",
  });
}
