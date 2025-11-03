import { NextRequest, NextResponse } from "next/server";
import { PrismaClient } from "@prisma/client";
import twilio from "twilio";

const prisma = new PrismaClient();

export async function POST(req: NextRequest) {
  try {
    const formData = await req.formData();
    const callSid = formData.get("CallSid") as string | null;
  const callStatus = formData.get("CallStatus") as string | null; // queued|ringing|in-progress|completed|busy|failed|no-answer|canceled
    const answeredByCb = (formData.get("AnsweredBy") as string | null) || null;
  const callDurationStr = (formData.get("CallDuration") as string | null) || null; // seconds as string

    console.log(`[Call Status] CallSid=${callSid} CallStatus=${callStatus} AnsweredBy=${answeredByCb}`);
    console.log(`[Call Status] Keys: ${Array.from(formData.keys()).join(", ")}`);

    if (callSid) {
      let statusLabel = "In Progress";
      switch (callStatus) {
        case "queued":
          statusLabel = "Queued";
          break;
        case "ringing":
          statusLabel = "Ringing";
          break;
        case "in-progress":
          statusLabel = "In Progress";
          break;
        case "completed":
          statusLabel = "Completed";
          break;
        case "busy":
          statusLabel = "Busy";
          break;
        case "no-answer":
          statusLabel = "No Answer";
          break;
        case "failed":
          statusLabel = "Failed";
          break;
        case "canceled":
          statusLabel = "Canceled";
          break;
      }

      // Update general status first
      const dataUpdate: any = { status: statusLabel };
      if (callDurationStr && !isNaN(Number(callDurationStr))) {
        dataUpdate.duration = Number(callDurationStr);
      }
      await prisma.callLog.updateMany({ where: { callSid }, data: dataUpdate });

      // If Twilio provided AnsweredBy in this callback, use it to set AMD result immediately
      if (answeredByCb) {
        const amdResult = answeredByCb === "human" ? "human" : answeredByCb.startsWith("machine") ? "machine" : null;
        if (amdResult) {
          await prisma.callLog.updateMany({ where: { callSid }, data: { result: amdResult, status: amdResult === "human" ? "Human Detected" : "Machine Detected" } });
        }
      }

      // On completion, as a fallback, fetch Call resource to read answeredBy when async AMD callback didn't arrive
      if (callStatus === "completed") {
        try {
          const callLog = await prisma.callLog.findFirst({ where: { callSid } });
          if (callLog && callLog.strategy === "Twilio AMD" && !callLog.result) {
            const client = twilio(process.env.TWILIO_ACCOUNT_SID, process.env.TWILIO_AUTH_TOKEN);
            const call = await client.calls(callSid!).fetch();
            // @ts-ignore answeredBy exists on the CallInstance when AMD enabled
            const answeredBy = (call as any).answeredBy as string | undefined;
            // @ts-ignore duration may be present on completed calls
            const fetchedDuration = (call as any).duration as number | undefined;
            if (answeredBy) {
              const amdResult = answeredBy === "human" ? "human" : answeredBy.startsWith("machine") ? "machine" : null;
              if (amdResult) {
                await prisma.callLog.updateMany({
                  where: { callSid },
                  data: { result: amdResult, status: amdResult === "human" ? "Human Detected" : "Machine Detected", ...(fetchedDuration ? { duration: Number(fetchedDuration) } : {}) },
                });
                console.log(`[Call Status] Fallback fetched answeredBy=${answeredBy} and updated AMD result for ${callSid}`);
              } else {
                console.log(`[Call Status] Fallback answeredBy not conclusive: ${answeredBy}`);
              }
            } else {
              console.log(`[Call Status] Fallback fetch had no answeredBy for ${callSid}`);
            }
          }
        } catch (e) {
          console.warn(`[Call Status] Fallback Twilio fetch failed for ${callSid}:`, e);
        }
      }
    }

    return NextResponse.json({ ok: true });
  } catch (e) {
    console.error("Error in call-status:", e);
    return NextResponse.json({ error: "call-status failed" }, { status: 500 });
  } finally {
    await prisma.$disconnect();
  }
}
