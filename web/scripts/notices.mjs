import { readFile, readdir, writeFile } from "node:fs/promises";
import { join } from "node:path";
const lock = JSON.parse(await readFile("package-lock.json", "utf8"));
const notices = [
  "Infinidev Web — third-party notices",
  "Generated from the production dependency lockfile.\n",
];
for (const [path, pkg] of Object.entries(lock.packages)) {
  if (!path || pkg.dev) continue;
  notices.push(
    `\n${path.replace(/^node_modules\//, "")} ${pkg.version} — ${pkg.license || "See license below"}\n`,
  );
  for (const file of await readdir(path)) {
    if (/^(licen[cs]e|copying|notice|ofl)(\.|$)/i.test(file)) {
      try {
        notices.push(await readFile(join(path, file), "utf8"));
      } catch (error) {
        if (error.code !== "EISDIR") throw error;
      }
    }
  }
}
await writeFile(
  "../src/infinidev/server/static/THIRD-PARTY-NOTICES.txt",
  notices.join("\n"),
);
