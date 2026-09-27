#!/usr/bin/env node
// Run the shipped upload handlers with browser FormData and a recorded HTTP boundary.
// The TensorAgent test project also exercises its full page through JavaScriptCore.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import vm from 'node:vm';

const repo = new URL('../../', import.meta.url);
const desktop = readFileSync(new URL('TensorSharp.Server.Host/wwwroot/index.html', repo), 'utf8');
const agent = readFileSync(new URL('TensorAgent/src/TensorAgent.Core/WebUi/tensoragent.js', repo), 'utf8');

function section(source, start, end) {
  const first = source.indexOf(start);
  const last = source.indexOf(end, first);
  assert.ok(first >= 0 && last > first, `Missing shipped handler: ${start}`);
  return source.slice(first, last);
}

function browser(kind, reply) {
  const calls = [], errors = [], warnings = [];
  const input = { value: '', files: [], addEventListener(name, action) { this[name] = action; } };
  let paints = 0;
  const context = vm.createContext({
    FormData, Promise, Array,
    fetch: async (path, options) => {
      calls.push({ path, ...options });
      const answer = typeof reply === 'function' ? await reply(calls.length) : reply;
      if (answer instanceof Error) throw answer;
      return { ok: true, json: async () => answer, ...answer?.http };
    },
    $: () => input,
    state: { attachments: [] },
    pendingAttachments: [],
    paintChips: () => { paints++; },
    renderAttachments: () => { paints++; },
    notice: (message, kind) => (kind === 'error' ? errors : warnings).push(message),
    alert: message => (/^Upload error:/.test(message) ? errors : warnings).push(message),
  });
  if (kind === 'desktop') {
    vm.runInContext(section(desktop, 'let uploadQueue =', '// The file name is the client'), context);
    input.change = context.handleFileSelect;
  } else {
    vm.runInContext(section(agent, '  var uploadQueue =', '  // ---- sheets'), context);
  }
  return {
    context, input, calls, errors, warnings,
    get attachments() { return kind === 'desktop' ? context.pendingAttachments : context.state.attachments; },
    get paints() { return paints; },
    async select(files) {
      input.files = files;
      input.value = 'selected';
      await input.change({ target: input });
      // The mobile event handler starts a promise chain without returning it.
      await new Promise(setImmediate);
    },
  };
}

const uploaded = [
  { ok: true, file: 'one.png', fileName: 'same.png', mediaType: 'image' },
  { ok: true, file: 'two.png', fileName: 'same.png', mediaType: 'image' },
  { ok: true, file: 'notes.txt', fileName: 'notes.txt', mediaType: 'text', textContent: 'SECONDARY EVIDENCE' },
  { ok: true, file: 'rows.csv', fileName: 'rows.csv', mediaType: 'text', fileBacked: true },
];
const files = uploaded.map((file, index) => new File([`bytes-${index}`], file.fileName));

for (const kind of ['desktop', 'agent']) {
  test(`${kind}: mixed files and duplicate names make one ordered multipart request`, async () => {
    const page = browser(kind, { ok: true, files: uploaded });
    await page.select(files);
    assert.equal(page.calls.length, 1);
    assert.equal(page.calls[0].path, '/api/upload');
    assert.equal(page.calls[0].method, 'POST');
    const parts = [...page.calls[0].body.entries()];
    assert.deepEqual(parts.map(([field, file]) => [field, file.name]), files.map(file => ['file', file.name]));
    assert.deepEqual(await Promise.all(parts.map(([, file]) => file.text())), files.map((_, index) => `bytes-${index}`));
    assert.deepEqual(page.attachments, uploaded);
    assert.equal(page.paints, 1);
    assert.equal(page.input.value, '');
    assert.deepEqual(page.errors, []);
  });

  test(`${kind}: one selected file accepts the legacy response and warning`, async () => {
    const result = { ...uploaded[0], warning: 'Preview unavailable' };
    const page = browser(kind, result);
    await page.select(files.slice(0, 1));
    assert.deepEqual(page.attachments, [result]);
    assert.deepEqual(page.warnings, ['Preview unavailable']);
    assert.deepEqual(page.errors, []);
  });

  for (const [name, reply, expectedError] of [
    ['HTTP rejection', { error: 'Quota exceeded', http: { ok: false } }, /Quota exceeded/],
    ['network rejection', new Error('Connection lost'), /Connection lost/],
    ['legacy server returning only one file', uploaded[0], /every uploaded file/],
    ['incomplete batch', { ok: true, files: [uploaded[0]] }, /every uploaded file/],
    ['failed batch item', { ok: true, files: [uploaded[0], { ok: false }] }, /every uploaded file/],
  ]) {
    test(`${kind}: ${name} keeps existing attachments without adding a partial batch`, async () => {
      const page = browser(kind, reply);
      page.attachments.push(uploaded[2]);
      await page.select(files.slice(0, 2));
      assert.equal(page.calls.length, 1);
      assert.deepEqual(page.attachments, [uploaded[2]]);
      assert.equal(page.paints, 0);
      assert.equal(page.errors.length, 1);
      assert.match(page.errors[0], expectedError);
    });
  }

  test(`${kind}: cancelled selection makes no upload request`, async () => {
    const page = browser(kind, {});
    await page.select([]);
    assert.equal(page.calls.length, 0);
    assert.equal(page.attachments.length, 0);
  });

  test(`${kind}: overlapping selections remain in pick order and recover after a failed upload`, async () => {
    let release;
    const page = browser(kind, call => call === 1
      ? new Promise(resolve => { release = resolve; })
      : uploaded[1]);
    const first = page.select(files.slice(0, 1));
    await new Promise(setImmediate);
    const second = page.select(files.slice(1, 2));
    await new Promise(setImmediate);
    assert.equal(page.calls.length, 1, 'the second selection must wait for the first');
    assert.equal(vm.runInContext('pendingUploadCount', page.context), 2);
    release({ error: 'First upload failed', http: { ok: false } });
    await Promise.all([first, second]);
    assert.equal(page.calls.length, 2);
    assert.deepEqual(page.attachments, [uploaded[1]]);
    assert.match(page.errors[0], /First upload failed/);
    assert.equal(vm.runInContext('pendingUploadCount', page.context), 0);
  });
}

test('desktop: every uploaded file reaches one chat turn in selection order', async () => {
  const page = browser('desktop', { ok: true, files: uploaded });
  await page.select(files);
  Object.assign(page.context, {
    isGenerating: false,
    currentLoadedModel: 'test-model',
    messageInput: { value: 'Compare these files', style: {} },
    chatHistory: [],
    addUserBubble() {},
    attachmentsDiv: { innerHTML: '' },
    chatContainer: { querySelector: () => null },
    requestAssistantResponse: async () => {},
  });
  vm.runInContext(section(desktop, 'async function sendMessage()', 'async function runImageEdit('), page.context);
  await page.context.sendMessage();
  const [message] = JSON.parse(JSON.stringify(page.context.chatHistory));
  assert.deepEqual(message.imagePaths, ['one.png', 'two.png']);
  assert.deepEqual(message.stillImagePaths, ['one.png', 'two.png']);
  assert.deepEqual(message.textFilePaths, ['notes.txt', 'rows.csv']);
  assert.deepEqual(message.textFileNames, ['notes.txt', 'rows.csv']);
  assert.match(message.content, /SECONDARY EVIDENCE/);
  assert.deepEqual(message.attachments.map(file => file.file), uploaded.map(file => file.file));
  assert.equal(message.attachments[3].fileBacked, true);
  assert.equal(page.attachments.length, 0);
});

test('desktop: Send while uploading preserves the draft and waits for every selected batch', async () => {
  let release;
  const page = browser('desktop', call => call === 1
    ? new Promise(resolve => { release = resolve; })
    : uploaded[2]);
  let generations = 0;
  Object.assign(page.context, {
    isGenerating: false,
    currentLoadedModel: 'test-model',
    messageInput: { value: 'Use every file', style: {} },
    chatHistory: [],
    addUserBubble() {},
    attachmentsDiv: { innerHTML: '' },
    chatContainer: { querySelector: () => null },
    requestAssistantResponse: async () => { generations++; },
  });
  vm.runInContext(section(desktop, 'async function sendMessage()', 'async function runImageEdit('), page.context);
  const first = page.select(files.slice(0, 2));
  await new Promise(setImmediate);
  const second = page.select(files.slice(2, 3));
  await page.context.sendMessage();
  assert.equal(generations, 0);
  assert.equal(page.context.messageInput.value, 'Use every file');
  assert.equal(page.context.chatHistory.length, 0);
  assert.match(page.warnings[0], /wait for file uploads/);
  release({ ok: true, files: uploaded.slice(0, 2) });
  await Promise.all([first, second]);
  assert.deepEqual(page.attachments, uploaded.slice(0, 3));
  await page.context.sendMessage();
  assert.equal(generations, 1);
  assert.deepEqual(Array.from(page.context.chatHistory[0].attachments, a => a.file), ['one.png', 'two.png', 'notes.txt']);
});
