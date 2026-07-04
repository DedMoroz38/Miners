// Образец шлифа в очереди лаборатории.
export type Sample = {
  id: string;
  name: string;
  meta: string; // условия съёмки / масштаб
  seed: number; // детерминированность мок-инференса
  bias: "common" | "thin" | "talc";
  imageUrl?: string; // object URL загруженного снимка (если есть)
  serverId?: string; // id образца на бэкенде — есть у реально загруженных
  tilesUrl?: string; // DZI-дескриптор для OpenSeadragon (гигапиксельный зум)
};

export type FileError = "type" | "empty";
