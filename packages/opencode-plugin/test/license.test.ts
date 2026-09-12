import { expect, test } from "bun:test";
import { readFile } from "node:fs/promises";
import { join } from "node:path";

const UPSTREAM_BSD3 = `BSD 3-Clause License

Copyright (c) 2026, Kevin Liao @ Aerovato Research

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions, and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
`;

test("ships a no-grant license plus verbatim upstream BSD-3 with the package", async () => {
  const rootLicense = await readFile(
    join(import.meta.dir, "../../../LICENSE.md"),
    "utf8",
  );
  const packagedLicense = await readFile(
    join(import.meta.dir, "../LICENSE.md"),
    "utf8",
  );
  const rootNotice = await readFile(
    join(import.meta.dir, "../../../NOTICE.md"),
    "utf8",
  );
  const packagedNotice = await readFile(
    join(import.meta.dir, "../NOTICE.md"),
    "utf8",
  );
  const packagedReadme = await readFile(
    join(import.meta.dir, "../README.md"),
    "utf8",
  );

  expect(packagedLicense).toBe(rootLicense);
  expect(packagedNotice).toBe(rootNotice);
  expect(rootLicense).toContain(
    "No license is granted for modifications, packaging, or other original material in this repository.",
  );
  expect(rootLicense).toContain("NOTICE.md");
  expect(rootLicense).not.toContain(
    "Redistribution and use in source and binary forms",
  );
  expect(rootNotice).toBe(UPSTREAM_BSD3);
  expect(rootNotice).not.toContain("Ivan Marshall Widjaja");

  const packageJson = JSON.parse(
    await readFile(join(import.meta.dir, "../package.json"), "utf8"),
  ) as { description: string; files: string[]; license: string };
  expect(packageJson.license).toBe("SEE LICENSE IN LICENSE.md");
  expect(packageJson.description).toContain(
    "Lossless persisted compaction for OpenCode V1",
  );
  expect(packageJson.description).toContain(
    "request-time tool-output projection for V2",
  );
  expect(packageJson.description).toContain("OpenCode 2.0.2");
  expect(packageJson.files).toContain("README.md");
  expect(packageJson.files).toContain("LICENSE.md");
  expect(packageJson.files).toContain("NOTICE.md");
  expect(packagedReadme).toContain("OpenCode V1");
  expect(packagedReadme).toContain("OpenCode V2 (validated against 2.0.2)");
});
