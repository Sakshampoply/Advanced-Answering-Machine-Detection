-- DropForeignKey
ALTER TABLE "public"."CallLog" DROP CONSTRAINT "CallLog_userId_fkey";

-- AlterTable
ALTER TABLE "CallLog" ALTER COLUMN "userId" DROP NOT NULL;

-- AddForeignKey
ALTER TABLE "CallLog" ADD CONSTRAINT "CallLog_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE SET NULL ON UPDATE CASCADE;
