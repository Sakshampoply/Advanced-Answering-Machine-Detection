"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";

export default function DialForm() {
  const [phone, setPhone] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const router = useRouter();

  const handleDial = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setSuccess(null);

    try {
      // Validate phone number
      if (!phone.trim()) {
        setError("Please enter a phone number");
        return;
      }

      const response = await fetch("/api/dial", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          phone,
          // userId will be added later when authentication is implemented
        }),
      });

      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.error || "Failed to initiate call");
      }

      const data = await response.json();
      setSuccess(`Call initiated! Call SID: ${data.callSid}`);
      setPhone("");

      // Redirect to history after 2 seconds
      setTimeout(() => {
        router.push("/history");
      }, 2000);
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="w-full max-w-md mx-auto p-6 bg-white rounded-lg shadow-md">
      <h1 className="text-2xl font-bold mb-6 text-center">AMD Dialer</h1>

      <form onSubmit={handleDial} className="space-y-4">
        <div>
          <label htmlFor="phone" className="block text-sm font-medium mb-2">
            Phone Number
          </label>
          <input
            id="phone"
            type="tel"
            placeholder="+1 (800) 123-4567"
            value={phone}
            onChange={(e) => setPhone(e.target.value)}
            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            disabled={loading}
          />
          <p className="text-xs text-gray-500 mt-1">
            💡 Tip: For trial accounts, use a verified number or test with <code className="bg-gray-100 px-1 rounded">+15005550006</code>
          </p>
        </div>

        <div>
          <label htmlFor="strategy" className="block text-sm font-medium mb-2">
            AMD Strategy
          </label>
          <select
            id="strategy"
            disabled
            className="w-full px-4 py-2 border border-gray-300 rounded-lg bg-gray-100 text-gray-600 cursor-not-allowed"
          >
            <option>Gemini Flash 2.5 Live API</option>
          </select>
          <p className="text-xs text-gray-500 mt-1">
            Currently using Gemini Flash only
          </p>
        </div>

        {error && (
          <div className="p-3 bg-red-100 border border-red-400 text-red-700 rounded">
            {error}
          </div>
        )}

        {success && (
          <div className="p-3 bg-green-100 border border-green-400 text-green-700 rounded">
            {success}
          </div>
        )}

        <button
          type="submit"
          disabled={loading}
          className="w-full px-4 py-2 bg-blue-600 text-white font-semibold rounded-lg hover:bg-blue-700 disabled:bg-gray-400 transition"
        >
          {loading ? "Dialing..." : "Dial Now"}
        </button>
      </form>
    </div>
  );
}
