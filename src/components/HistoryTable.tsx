"use client";

import { useEffect, useState } from "react";

interface CallLogEntry {
  id: number;
  phone: string;
  strategy: string;
  status: string;
  result: string | null;
  confidence: number | null;
  duration: number | null;
  createdAt: string;
}

export default function HistoryTable() {
  const [calls, setCalls] = useState<CallLogEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchCalls = async () => {
      try {
        const response = await fetch("/api/call-logs?page=1&limit=50", {
          method: "GET",
          headers: { "Content-Type": "application/json" },
        });

        if (!response.ok) {
          throw new Error("Failed to fetch call logs");
        }

        const data = await response.json();
        setCalls(data.calls || []);
      } catch (err) {
        setError(err instanceof Error ? err.message : "An error occurred");
      } finally {
        setLoading(false);
      }
    };

    fetchCalls();
  }, []);

  if (loading) {
    return <div className="text-center py-8">Loading...</div>;
  }

  if (error) {
    return (
      <div className="p-4 bg-red-100 border border-red-400 text-red-700 rounded">
        {error}
      </div>
    );
  }

  return (
    <div className="w-full overflow-x-auto">
      <table className="min-w-full border-collapse border border-gray-300">
        <thead className="bg-gray-200">
          <tr>
            <th className="border border-gray-300 px-4 py-2 text-left">
              Phone Number
            </th>
            <th className="border border-gray-300 px-4 py-2 text-left">
              Strategy
            </th>
            <th className="border border-gray-300 px-4 py-2 text-left">
              Status
            </th>
            <th className="border border-gray-300 px-4 py-2 text-left">
              Result
            </th>
            <th className="border border-gray-300 px-4 py-2 text-left">
              Confidence
            </th>
            <th className="border border-gray-300 px-4 py-2 text-left">
              Duration
            </th>
            <th className="border border-gray-300 px-4 py-2 text-left">
              Date
            </th>
          </tr>
        </thead>
        <tbody>
          {calls.length === 0 ? (
            <tr>
              <td colSpan={7} className="border border-gray-300 px-4 py-2 text-center text-gray-500">
                No calls yet
              </td>
            </tr>
          ) : (
            calls.map((call) => (
              <tr key={call.id} className="hover:bg-gray-100">
                <td className="border border-gray-300 px-4 py-2">{call.phone}</td>
                <td className="border border-gray-300 px-4 py-2">
                  {call.strategy}
                </td>
                <td className="border border-gray-300 px-4 py-2">
                  <span
                    className={`px-2 py-1 rounded text-sm font-semibold ${
                      call.status === "Human Detected"
                        ? "bg-green-200 text-green-800"
                        : call.status === "Machine Detected"
                          ? "bg-red-200 text-red-800"
                          : "bg-yellow-200 text-yellow-800"
                    }`}
                  >
                    {call.status}
                  </span>
                </td>
                <td className="border border-gray-300 px-4 py-2">
                  {call.result || "-"}
                </td>
                <td className="border border-gray-300 px-4 py-2">
                  {call.confidence ? `${(call.confidence * 100).toFixed(1)}%` : "-"}
                </td>
                <td className="border border-gray-300 px-4 py-2">
                  {call.duration ? `${call.duration}s` : "-"}
                </td>
                <td className="border border-gray-300 px-4 py-2">
                  {new Date(call.createdAt).toLocaleString()}
                </td>
              </tr>
            ))
          )}
        </tbody>
      </table>
    </div>
  );
}
