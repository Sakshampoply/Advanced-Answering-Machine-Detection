import HistoryTable from "@/components/HistoryTable";
import Link from "next/link";

export default function HistoryPage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-8">
      <div className="max-w-6xl mx-auto">
        <div className="flex justify-between items-center mb-8">
          <h1 className="text-3xl font-bold text-gray-800">Call History</h1>
          <Link
            href="/"
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition"
          >
            Back to Dialer
          </Link>
        </div>

        <div className="bg-white rounded-lg shadow-md p-6">
          <HistoryTable />
        </div>
      </div>
    </div>
  );
}
