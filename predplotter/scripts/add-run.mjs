#!/usr/bin/env node
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const projectRoot = path.resolve(__dirname, "..");
const csvPath = path.join(projectRoot, "public", "progress.csv");

const args = process.argv.slice(2);
let label = "Kaggle run";
for (let i = 0; i < args.length; i += 1) {
  if (args[i] === "--label" && args[i + 1]) {
    label = args[i + 1];
    i += 1;
  } else if (args[i] === "--help" || args[i] === "-h") {
    printHelp();
    process.exit(0);
  }
}

const stdin = await readStdin();
if (!stdin.trim()) {
  console.error("No log content detected on stdin.");
  process.exit(1);
}

const { fullCv, processedCv } = extractScores(stdin);
if (fullCv === null || processedCv === null) {
  console.error("Unable to locate CV scores in the provided log.");
  process.exit(1);
}

ensureCsvExists();
const fileRaw = fs.readFileSync(csvPath, "utf8");
const trimmed = fileRaw.trim();
const rows = trimmed ? trimmed.split(/\r?\n/) : [];
let nextId = 1;
if (rows.length > 1) {
  const last = rows.at(-1);
  if (last && !last.startsWith("run_id")) {
    const [id] = last.split(",");
    nextId = Number(id) + 1;
  }
}

const timestamp = new Date().toISOString();
const newRow = `${nextId},${timestamp},"${escapeLabel(label)}",${fullCv.toFixed(
  6,
)},${processedCv.toFixed(6)}`;
const needsNewline = fileRaw.length > 0 && !fileRaw.endsWith("\n");
const prefix = needsNewline ? "\n" : "";
fs.appendFileSync(csvPath, `${prefix}${newRow}\n`);
console.log(
  `Appended run #${nextId}: full=${fullCv.toFixed(
    3,
  )} processed=${processedCv.toFixed(3)}`,
);

function extractScores(logText) {
  const fullMatches = [...logText.matchAll(/Full CV Score:\s*([\d.]+)/g)];
  const procMatches = [...logText.matchAll(/Processed CV Score:\s*([\d.]+)/g)];
  const full = fullMatches.at(-1)?.[1] ?? null;
  const proc = procMatches.at(-1)?.[1] ?? null;
  return {
    fullCv: full ? Number(full) : null,
    processedCv: proc ? Number(proc) : null,
  };
}

function ensureCsvExists() {
  if (!fs.existsSync(csvPath)) {
    fs.mkdirSync(path.dirname(csvPath), { recursive: true });
    fs.writeFileSync(
      csvPath,
      "run_id,timestamp,label,full_cv,processed_cv\n",
      "utf8",
    );
  }
}

function escapeLabel(str) {
  return str.replace(/"/g, '""');
}

function printHelp() {
  console.log(`Usage: npm run add-run -- --label "description" < kaggle.log

Reads Kaggle notebook output from stdin, finds the last reported Full/Processed CV scores,
and appends a new row to public/progress.csv so the dashboard can visualize it.

Options:
  --label <text>   Label to store alongside the run (defaults to "Kaggle run")
  -h, --help       Show this help text
`);
}

function readStdin() {
  return new Promise((resolve) => {
    let data = "";
    process.stdin.setEncoding("utf8");
    process.stdin.on("data", (chunk) => {
      data += chunk;
    });
    process.stdin.on("end", () => resolve(data));
    if (process.stdin.isTTY) resolve("");
  });
}
