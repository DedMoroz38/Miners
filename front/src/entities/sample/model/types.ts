// Образец шлифа в очереди лаборатории.
export type Sample = {
  id: string;
  name: string;
  meta: string; // условия съёмки / масштаб
  seed: number; // детерминированность мок-инференса
  bias: "common" | "thin" | "talc";
  imageUrl?: string; // object URL загруженного снимка (если есть)
};

export type FileError = "type" | "empty";
