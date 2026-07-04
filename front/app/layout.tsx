import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "ШЛИФ — Lab Console",
  description:
    "Лабораторная консоль автоматической классификации руд по панорамным микрофотографиям полированных шлифов",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="ru">
      <body>{children}</body>
    </html>
  );
}
