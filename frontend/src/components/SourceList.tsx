import type { Source } from "../api/client";

interface Props {
  sources: Source[];
}

export default function SourceList({ sources }: Props) {
  if (!sources.length) return null;

  return (
    <div className="sources">
      <div className="sources-label">Джерела:</div>
      <ul className="sources-list">
        {sources.map((s, i) => (
          <li key={i} className="source-item">
            <span className="source-title">{s.title}</span>
            {s.section && <span className="source-section"> › {s.section}</span>}
            <span className="source-relevance">{(s.relevance * 100).toFixed(0)}%</span>
          </li>
        ))}
      </ul>
    </div>
  );
}
