/*
  Warnings:

  - A unique constraint covering the columns `[callSid]` on the table `CallLog` will be added. If there are existing duplicate values, this will fail.

*/
-- AlterTable
ALTER TABLE "CallLog" ADD COLUMN     "callSid" TEXT;

-- CreateIndex
CREATE UNIQUE INDEX "CallLog_callSid_key" ON "CallLog"("callSid");
