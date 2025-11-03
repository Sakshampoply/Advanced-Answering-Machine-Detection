import { NextRequest, NextResponse } from "next/server";
import { PrismaClient } from "@prisma/client";

const prisma = new PrismaClient();

export async function POST(req: NextRequest) {
  try {
    const formData = await req.formData();
    const callSid = formData.get("CallSid") as string | null;
    const machineDetectionStatus = formData.get(
      "AnsweringMachineDetectionStatus"
    ) as string | null;
    // Some Twilio accounts/regions send AnsweredBy instead of AnsweringMachineDetectionStatus
    const answeredBy = (formData.get("AnsweredBy") as string | null) || null;
    const callStatus = formData.get("CallStatus") as string | null;

    console.log(
      `AMD Status - Call: ${callSid}, MachineDetectionStatus: ${machineDetectionStatus}, AnsweredBy: ${answeredBy}, CallStatus: ${callStatus}`
    );
    console.log(`[AMD Status] Keys: ${Array.from(formData.keys()).join(", ")}`);

    // If neither AMD field is present, ignore gracefully
    if (!machineDetectionStatus && !answeredBy) {
      console.log("[AMD Status] No AMD fields present; ignoring non-AMD callback.");
      return NextResponse.json({ message: "Ignored non-AMD callback" }, { status: 200 });
    }

    // Map Twilio AMD status to our status
    let status = "Unknown";
    let result = null;

    if (machineDetectionStatus === "human_detected" || answeredBy === "human") {
      status = "Human Detected";
      result = "human";
    } else if (
      machineDetectionStatus === "machine_detected" ||
      machineDetectionStatus === "machine_start" ||
      answeredBy?.startsWith("machine")
    ) {
      status = "Machine Detected";
      result = "machine";
    } else if (callStatus === "no-answer") {
      status = "No Answer";
      result = "no_answer";
    } else if (callStatus === "busy") {
      status = "Busy";
      result = "busy";
    } else if (callStatus === "completed") {
      status = "Completed";
      result = result || "completed";
    }

    // Update call log in database by callSid
    if (callSid) {
      const updated = await prisma.callLog.updateMany({
        where: { callSid },
        data: { status, result },
      });

      console.log(
        `Updated ${updated.count} call(s) with status: ${status}, result: ${result}`
      );
    } else {
      console.warn("No CallSid provided in AMD status callback");
    }

    return NextResponse.json({ message: "Status updated" }, { status: 200 });
  } catch (error) {
    console.error("Error processing AMD status:", error);
    return NextResponse.json(
      { error: "Failed to process AMD status" },
      { status: 500 }
    );
  }
}
