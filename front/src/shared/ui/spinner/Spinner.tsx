export function Spinner({ className = "" }: { className?: string }) {
  return (
    <span
      className={`inline-block animate-spin rounded-full border-2 border-ink/30 border-t-ink ${className || "h-4 w-4"}`}
    />
  );
}
