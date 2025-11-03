import { NextRequest, NextResponse } from "next/server";
import { PrismaClient } from "@prisma/client";

const prisma = new PrismaClient();

export async function GET(req: NextRequest) {
  try {
    // Get pagination params from query
    const page = parseInt(req.nextUrl.searchParams.get("page") || "1");
    const limit = parseInt(req.nextUrl.searchParams.get("limit") || "10");
    const skip = (page - 1) * limit;

    // Fetch call logs from database (all calls, no user filter for now)
    const callLogs = await prisma.callLog.findMany({
      skip,
      take: limit,
      orderBy: { createdAt: "desc" },
      select: {
        id: true,
        phone: true,
        strategy: true,
        status: true,
        result: true,
        confidence: true,
        duration: true,
        createdAt: true,
      },
    });

    // Get total count for pagination
    const total = await prisma.callLog.count();

    return NextResponse.json(
      {
        calls: callLogs,
        pagination: {
          page,
          limit,
          total,
          pages: Math.ceil(total / limit),
        },
      },
      { status: 200 }
    );
  } catch (error) {
    console.error("Error fetching call logs:", error);
    return NextResponse.json(
      { error: "Failed to fetch call logs" },
      { status: 500 }
    );
  }
}
