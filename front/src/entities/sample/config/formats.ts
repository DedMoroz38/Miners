// Форматы панорамных снимков, которые принимает лаборатория.
export const ACCEPTED_TYPES = [
  "image/tiff",
  "image/png",
  "image/jpeg",
] as const;

export const ACCEPTED_EXT = [".tif", ".tiff", ".png", ".jpg", ".jpeg"];

export const MAX_DIM = 10000; // px по стороне
