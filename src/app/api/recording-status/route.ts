import { NextRequest, NextResponse } from "next/server";

export async function POST(req: NextRequest) {
  try {
    const formData = await req.formData();
    const recordingSid = formData.get("RecordingSid") as string | null;
    const recordingStatus = formData.get("RecordingStatus") as string | null;
    const recordingUrl = formData.get("RecordingUrl") as string | null;

    console.log(
      `Recording ${recordingSid} - Status: ${recordingStatus}, URL: ${recordingUrl}`
    );

    return NextResponse.json({ message: "Recording processed" }, { status: 200 });
  } catch (error) {
    console.error("Error processing recording status:", error);
    return NextResponse.json(
      { error: "Failed to process recording status" },
      { status: 500 }
    );
  }
}
