'use client';

import {
  CategoryScale,
  Chart as ChartJS,
  Filler,
  Legend,
  LineElement,
  LinearScale,
  PointElement,
  Tooltip,
} from "chart.js";
import { Line } from "react-chartjs-2";
import { useEffect, useMemo, useState } from "react";

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Tooltip,
  Legend,
  Filler,
);

type RunEntry = {
  runId: string;
  timestamp: string;
  label: string;
  fullCv: number;
  processedCv: number;
};

const CSV_URL = "/progress.csv";

export default function Home() {
  const [runs, setRuns] = useState<RunEntry[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [logInput, setLogInput] = useState("");
  const [labelInput, setLabelInput] = useState("");
  const [formMessage, setFormMessage] = useState<string | null>(null);
  const [copyStatus, setCopyStatus] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetch(CSV_URL)
      .then((res) => {
        if (!res.ok) throw new Error(`Unable to load ${CSV_URL}`);
        return res.text();
      })
      .then((text) => {
        if (cancelled) return;
        const parsed = parseCsv(text);
        setRuns(parsed);
      })
      .catch((err) => {
        if (!cancelled) setError(err.message);
      });

    return () => {
      cancelled = true;
    };
  }, []);

  const sortedRuns = useMemo(() => {
    return [...runs].sort(
      (a, b) =>
        new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime(),
    );
  }, [runs]);

  const latest = sortedRuns.at(-1);

  const chartData = useMemo(() => {
    const labels = sortedRuns.map((run) => `#${run.runId}`);
    return {
      labels,
      datasets: [
        {
          label: "Full CV",
          data: sortedRuns.map((run) => run.fullCv),
          tension: 0.4,
          borderColor: "rgb(251 191 36)",
          backgroundColor: "rgba(251, 191, 36, 0.15)",
          fill: true,
          pointRadius: 4,
          pointBackgroundColor: "rgb(251 191 36)",
        },
        {
          label: "Processed CV",
          data: sortedRuns.map((run) => run.processedCv),
          tension: 0.45,
          borderColor: "rgb(16 185 129)",
          backgroundColor: "rgba(16, 185, 129, 0.2)",
          fill: true,
          pointRadius: 5,
          pointBackgroundColor: "rgb(16 185 129)",
        },
      ],
    };
  }, [sortedRuns]);

  const chartOptions = useMemo(
    () => ({
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          labels: { color: "#e2e8f0", boxWidth: 12 },
        },
        tooltip: {
          callbacks: {
            label(context) {
              return `${context.dataset.label}: ${Number(
                context.parsed.y,
              ).toFixed(3)}`;
            },
          },
        },
      },
      scales: {
        x: {
          ticks: { color: "#94a3b8" },
          grid: { color: "rgba(255,255,255,0.05)" },
        },
        y: {
          ticks: { color: "#94a3b8" },
          grid: { color: "rgba(255,255,255,0.05)" },
          beginAtZero: false,
        },
      },
    }),
    [],
  );

  const handleAddRun = () => {
    setFormMessage(null);
    const trimmedLog = logInput.trim();
    if (!trimmedLog) {
      setFormMessage("Please paste the Kaggle output first.");
      return;
    }
    const extracted = extractFromLog(trimmedLog);
    if (!extracted) {
      setFormMessage("Could not find CV scores in the provided log.");
      return;
    }
    const nextId = `${computeNextId(runs)}`;
    const entry: RunEntry = {
      runId: nextId,
      timestamp: new Date().toISOString(),
      label: labelInput.trim() || `Run #${nextId}`,
      fullCv: extracted.fullCv,
      processedCv: extracted.processedCv,
    };
    setRuns((prev) => [...prev, entry]);
    setLogInput("");
    setLabelInput("");
    setFormMessage(
      `Added run #${nextId}: processed CV ${entry.processedCv.toFixed(3)}`,
    );
  };

  const handleCopyCsv = async () => {
    setCopyStatus(null);
    try {
      const csv = formatCsv(sortedRuns);
      await navigator.clipboard.writeText(csv);
      setCopyStatus("Copied updated CSV to clipboard.");
    } catch (err) {
      setCopyStatus("Unable to copy CSV. Copy manually from DevTools.");
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-50">
      <main className="mx-auto flex max-w-6xl flex-col gap-10 px-6 py-12">
        <header>
          <p className="text-xs uppercase tracking-[0.3em] text-slate-500">
            CSIRO I2B tracker
          </p>
          <h1 className="mt-2 text-3xl font-semibold">
            Offline Progress Plotter
          </h1>
          <p className="mt-2 text-base text-slate-300">
            Paste logs from Kaggle runs, capture the CV summary, and visualize
            the journey toward the 0.82 target. Use the controls below to append
            new entries and export an updated CSV for version control.
          </p>
        </header>

        {error && (
          <div className="rounded border border-rose-500/40 bg-rose-500/10 p-4 text-sm text-rose-200">
            {error}
          </div>
        )}

        <section className="rounded-xl border border-slate-800 bg-slate-900/40 p-6">
          <h2 className="text-lg font-semibold text-slate-100">
            Add Kaggle Output
          </h2>
          <p className="text-sm text-slate-400">
            Paste the raw notebook output; the parser grabs the last reported
            Full/Processed CV scores automatically.
          </p>
          <div className="mt-4 flex flex-col gap-4">
            <textarea
              value={logInput}
              onChange={(e) => setLogInput(e.target.value)}
              placeholder="Paste Kaggle logs here..."
              className="min-h-[140px] rounded-lg border border-slate-800 bg-slate-950/30 px-4 py-3 font-mono text-sm text-slate-100 placeholder:text-slate-600 focus:border-emerald-400 focus:outline-none focus:ring-1 focus:ring-emerald-400"
            />
            <div className="flex flex-col gap-3 md:flex-row md:items-center">
              <input
                value={labelInput}
                onChange={(e) => setLabelInput(e.target.value)}
                placeholder="Short label (optional)"
                className="w-full rounded-lg border border-slate-800 bg-slate-950/30 px-3 py-2 text-sm text-slate-100 placeholder:text-slate-600 focus:border-emerald-400 focus:outline-none focus:ring-1 focus:ring-emerald-400"
              />
              <button
                onClick={handleAddRun}
                className="w-full rounded-lg bg-emerald-500/90 px-4 py-2 text-sm font-semibold text-slate-950 transition hover:bg-emerald-400 md:w-auto"
              >
                Extract & Preview
              </button>
              <button
                onClick={handleCopyCsv}
                className="w-full rounded-lg border border-slate-700 px-4 py-2 text-sm font-semibold text-slate-200 transition hover:border-slate-500 hover:text-white md:w-auto"
                disabled={sortedRuns.length === 0}
              >
                Copy updated CSV
              </button>
            </div>
            {formMessage && (
              <p className="text-sm text-emerald-300">{formMessage}</p>
            )}
            {copyStatus && (
              <p className="text-sm text-slate-400">{copyStatus}</p>
            )}
          </div>
        </section>

        {sortedRuns.length > 0 ? (
          <>
            <section className="rounded-xl border border-slate-800 bg-slate-900/40 p-6">
              <div className="flex flex-col gap-1 md:flex-row md:items-center md:justify-between">
                <div>
                  <h2 className="text-lg font-semibold text-slate-100">
                    CV Trajectory
                  </h2>
                  <p className="text-sm text-slate-400">
                    Raw vs processed Full CV scores per run (Chart.js).
                  </p>
                </div>
                {latest && (
                  <p className="text-sm text-slate-400">
                    Latest processed CV:{" "}
                    <span className="font-semibold text-emerald-300">
                      {latest.processedCv.toFixed(3)}
                    </span>
                  </p>
                )}
              </div>
              <div className="mt-6 h-[340px]">
                <Line data={chartData} options={chartOptions} />
              </div>
            </section>

            <section className="rounded-xl border border-slate-800 bg-slate-900/40 p-6">
              <h2 className="text-lg font-semibold text-slate-100">
                Run Log ({sortedRuns.length})
              </h2>
              <div className="mt-4 overflow-x-auto">
                <table className="w-full min-w-[580px] text-left text-sm text-slate-200">
                  <thead>
                    <tr className="border-b border-slate-800/60 text-xs uppercase tracking-wide text-slate-400">
                      <th className="py-2 pr-4">Run</th>
                      <th className="py-2 pr-4">Timestamp</th>
                      <th className="py-2 pr-4">Label</th>
                      <th className="py-2 pr-4">Full CV</th>
                      <th className="py-2 pr-4">Processed CV</th>
                    </tr>
                  </thead>
                  <tbody>
                    {sortedRuns.map((run) => (
                      <tr
                        key={`${run.runId}-${run.timestamp}`}
                        className="border-b border-slate-800/40 last:border-0"
                      >
                        <td className="py-2 pr-4 font-mono text-xs text-slate-400">
                          #{run.runId}
                        </td>
                        <td className="py-2 pr-4 text-slate-300">
                          {new Date(run.timestamp).toLocaleString()}
                        </td>
                        <td className="py-2 pr-4">{run.label}</td>
                        <td className="py-2 pr-4 font-mono text-amber-200">
                          {run.fullCv.toFixed(3)}
                        </td>
                        <td className="py-2 pr-4 font-mono text-emerald-300">
                          {run.processedCv.toFixed(3)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>
          </>
        ) : (
          <p className="text-slate-400">
            No entries yet. Paste a Kaggle log to begin tracking progress.
          </p>
        )}
      </main>
    </div>
  );
}

function parseCsv(text: string): RunEntry[] {
  const rows = text
    .trim()
    .split(/\r?\n/)
    .filter(Boolean);
  if (rows.length <= 1) return [];
  const [, ...dataRows] = rows;
  return dataRows
    .map(parseCsvLine)
    .filter(
      (entry): entry is RunEntry =>
        !!entry && !Number.isNaN(entry.fullCv) && !Number.isNaN(entry.processedCv),
    );
}

function parseCsvLine(line: string): RunEntry | null {
  const parts: string[] = [];
  let current = "";
  let inQuotes = false;

  for (const char of line) {
    if (char === '"' ) {
      inQuotes = !inQuotes;
      continue;
    }
    if (char === "," && !inQuotes) {
      parts.push(current);
      current = "";
    } else {
      current += char;
    }
  }
  parts.push(current);

  if (parts.length < 5) return null;
  const [runId, timestamp, label, fullCv, processedCv] = parts;
  return {
    runId,
    timestamp,
    label,
    fullCv: Number(fullCv),
    processedCv: Number(processedCv),
  };
}

function extractFromLog(log: string): { fullCv: number; processedCv: number } | null {
  const fullMatches = [...log.matchAll(/Full CV Score:\s*([0-9.]+)/g)];
  const procMatches = [...log.matchAll(/Processed CV Score:\s*([0-9.]+)/g)];
  const full = fullMatches.at(-1)?.[1];
  const proc = procMatches.at(-1)?.[1];
  if (!full || !proc) return null;
  return { fullCv: Number(full), processedCv: Number(proc) };
}

function computeNextId(entries: RunEntry[]): number {
  if (entries.length === 0) return 1;
  return (
    entries.reduce((max, entry) => Math.max(max, Number(entry.runId)), 0) + 1
  );
}

function formatCsv(entries: RunEntry[]): string {
  const header = "run_id,timestamp,label,full_cv,processed_cv";
  const rows = entries.map(
    (entry) =>
      `${entry.runId},${entry.timestamp},"${entry.label.replace(/"/g, '""')}",${entry.fullCv},${entry.processedCv}`,
  );
  return [header, ...rows].join("\n");
}
