// Submission data never comes from a display label or an assumed frame rate.
const frameMetadataByImage = new Map();

function normalizePinnedFrame(value) {
  if (!value || typeof value !== 'object') throw new Error('Thiếu dữ liệu frame. Hãy tìm và ghim lại frame.');
  const videoId = String(value.videoId || '').trim().replace(/\.(mp4|mkv|avi|webm)$/i, '');
  const keyframeName = String(value.keyframeName ?? '').trim();
  const time = value.timeSeconds;
  if (!videoId || !keyframeName || time === null || time === undefined ||
      String(time).trim() === '' || !Number.isFinite(Number(time)) || Number(time) < 0) {
    throw new Error('Frame thiếu video, keyframe hoặc thời gian hợp lệ. Hãy tải lại dữ liệu và ghim lại.');
  }
  const timeSeconds = Number(time);
  if (!Number.isSafeInteger(Math.round(timeSeconds * 1000))) throw new Error('Thời gian frame vượt giới hạn.');
  const frameId = value.frameId == null ? null : Number(value.frameId);
  if (frameId !== null && (!Number.isSafeInteger(frameId) || frameId < 0)) throw new Error('Số frame không hợp lệ.');
  const key = JSON.stringify([videoId, keyframeName, timeSeconds]);
  return { videoId, keyframeName, frameId, timeSeconds, key,
    src: String(value.src || ''), videoFramePart: `${videoId}/${keyframeName}`,
    frameInfo: `${videoId}-${timeSeconds}` };
}

function frameFromResult(entity, src) {
  return { videoId: entity.video, keyframeName: entity.keyframe_name ?? entity.frame_id,
    frameId: entity.original_frame_id ?? null, timeSeconds: entity.time, src };
}

function parseFrameMapping(csv) {
  const lines = csv.trim().split(/\r?\n/);
  const headers = lines.shift().replace(/^\uFEFF/, '').split(',').map(s => s.trim());
  const n = headers.indexOf('n'), pts = headers.indexOf('pts_time'), original = headers.indexOf('frame_idx');
  if (n < 0 || pts < 0) throw new Error('CSV thiếu cột n hoặc pts_time.');
  const mapping = {};
  for (const line of lines) {
    if (!line.trim()) continue;
    const fields = line.split(',').map(s => s.trim());
    if (!/^\d+$/.test(fields[n] || '') || !fields[pts]) continue;
    const timeSeconds = Number(fields[pts]);
    if (!Number.isFinite(timeSeconds) || timeSeconds < 0) continue;
    const frameId = original >= 0 && /^\d+$/.test(fields[original] || '') ? Number(fields[original]) : null;
    mapping[String(Number(fields[n]))] = { timeSeconds, frameId };
  }
  return mapping;
}

function imageMetadataKey(src) { return new URL(src, window.location.href).href; }

function registerFrameMetadata(img, frame) {
  // Keep incomplete frames viewable; pinning/submission will explain the missing data.
  img.frameData = frame;
  frameMetadataByImage.set(imageMetadataKey(img.src), frame);
}

function frameFromImage(img) {
  if (!img) throw new Error('Không tìm thấy ảnh cần ghim.');
  return normalizePinnedFrame(img.frameData || frameMetadataByImage.get(imageMetadataKey(img.src)));
}

function validateQaAnswer(value) {
  const answer = String(value ?? '').trim();
  if (!/^[0-9]+$/.test(answer)) throw new Error('Đáp án QA phải là chuỗi chữ số 0–9 và không được để trống.');
  return answer; // Preserve leading zeroes, e.g. 007.
}

function buildKisQaBody(item, task, answer) {
  const frame = normalizePinnedFrame(item);
  const ms = Math.round(frame.timeSeconds * 1000);
  if (['qa', 'vqa', 'kis-qa'].includes(task)) {
    return { answerSets: [{ answers: [{ text: `${validateQaAnswer(answer)}-${frame.videoId}-${ms}` }] }] };
  }
  if (!['kis', 'kis-v'].includes(task)) throw new Error('Loại bài không được hỗ trợ.');
  return { answerSets: [{ answers: [{ mediaItemName: frame.videoId, start: ms, end: ms }] }] };
}
