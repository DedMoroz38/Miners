import { ACCEPTED_EXT, ACCEPTED_TYPES } from "../config/formats";
import type { FileError, Sample } from "../model/types";

export function validateFile(file: File): FileError | null {
  if (file.size === 0) return "empty";
  const okType = (ACCEPTED_TYPES as readonly string[]).includes(file.type);
  const okExt = ACCEPTED_EXT.some((e) => file.name.toLowerCase().endsWith(e));
  if (!okType && !okExt) return "type";
  return null;
}

// Образец из загруженного файла. bias/seed выводим детерминированно из имени,
// чтобы мок-инференс давал стабильный результат для одного и того же файла.
export function sampleFromFile(file: File): Sample {
  let hash = 0;
  for (let i = 0; i < file.name.length; i++)
    hash = (hash * 31 + file.name.charCodeAt(i)) | 0;
  const seed = Math.abs(hash) % 1000;
  const biases: Sample["bias"][] = ["common", "thin", "talc"];
  const bias = biases[Math.abs(hash) % 3];
  const sizeMb = (file.size / (1024 * 1024)).toFixed(1);
  const ext = file.name.split(".").pop()?.toUpperCase() ?? "IMG";
  return {
    id: `UP-${(Math.abs(hash) % 9000) + 1000}`,
    name: file.name.replace(/\.[^.]+$/, ""),
    meta: `Загружен · ${ext} · ${sizeMb} МБ`,
    seed,
    bias,
    imageUrl: URL.createObjectURL(file),
  };
}
