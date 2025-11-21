// SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
//
// SPDX-License-Identifier: Apache-2.0

// TODO: Polish after having the Haystack API endpoint
import React, {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { useHistory } from "@docusaurus/router";
import debounce from "lodash/debounce";
import styles from "./styles.module.css";

const MIN_QUERY_LENGTH = 3;
const DEBOUNCE_DELAY = 650;

const titleCase = (s) => {
  return s
    .toLowerCase()
    .split(/[\s_-]+/)
    .filter(Boolean)
    .map((w) => w[0]?.toUpperCase() + w.slice(1))
    .join(" ");
};

const toPlainText = (s) => {
  if (!s) return "";
  // Strip HTML tags
  let t = s.replace(/<[^>]+>/g, " ");
  // Convert markdown links [text](url) -> text
  t = t.replace(/\[([^\]]+)\]\([^)]+\)/g, "$1");
  // Replace markdown special characters with spaces
  t = t.replace(/[#>*_`~\-]+/g, " ");
  // Collapse whitespace
  t = t.replace(/\s+/g, " ").trim();
  return t;
};

const escapeRegExp = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

const buildSnippet = (content, query, maxLen = 200) => {
  const text = toPlainText(content);
  if (!text) return "";

  const terms = query.trim().split(/\s+/).filter(Boolean);
  if (terms.length === 0) {
    return text.length > maxLen ? text.slice(0, maxLen) + "…" : text;
  }

  const regex = new RegExp(`(${terms.map(escapeRegExp).join("|")})`, "ig");
  const match = regex.exec(text);
  const start = match ? Math.max(0, match.index - 60) : 0;
  const end = Math.min(text.length, start + maxLen);
  let slice = text.slice(start, end);
  // Only <mark> is injected; original HTML was stripped above
  let highlighted = slice.replace(regex, "<mark>$1</mark>");

  if (start > 0) highlighted = "… " + highlighted;

  if (end < text.length) highlighted += " …";

  return highlighted;
};

const extractTitle = (content, fileName, path) => {
  const h1 =
    content.match(/^#\s+(.+?)\s*$/m)?.[1] ||
    content.match(/^##\s+(.+?)\s*$/m)?.[1];

  if (h1) return toPlainText(h1);

  if (fileName) return titleCase(fileName.replace(/\.html?$/i, ""));

  if (path) {
    const last = path.split("/").filter(Boolean).pop() || path;
    return titleCase(last.replace(/\.html?$/i, ""));
  }

  return "Untitled";
};

const toDocUrl = (url) => {
  if (!url) return "/docs";

  let p = url;
  p = p.replace(/\/index\.html?$/i, "/");
  p = p.replace(/\.html?$/i, "");

  p = p.split("/docs").pop();

  if (!p.startsWith("/")) p = "/" + p;

  return "/docs" + p;
};

const groupByPage = (documents) => {
  const groups = new Map();

  for (const doc of documents) {
    const key =
      doc.meta?.original_file_path ||
      doc.meta?.url ||
      doc.file?.name ||
      "unknown";

    if (!groups.has(key)) groups.set(key, []);

    groups.get(key).push(doc);
  }

  return groups;
};

// TODO: Update this after having the Haystack API endpoint
const categorizeDocument = (doc, path) => {
  // First, try to use the navigation metadata from the document
  const navigation = doc?.meta?.type;
  if (navigation) {
    // Normalize the navigation value
    const normalized = navigation.toLowerCase().trim();
    if (normalized === "api-reference") {
      return "api-reference";
    }
    if (normalized === "documentation") {
      return "documentation";
    }
  }

  // Fall back to path-based categorization for documents without metadata
  if (!path) return "documentation";

  const lowerPath = path.toLowerCase();

  if (lowerPath.includes("/reference/")) {
    return "api-reference";
  }

  // Guides (how-tos, tutorials, learn) - default for most docs
  return "guides";
};

const toResults = (documents, query) => {
  const groups = groupByPage(documents);
  return Array.from(groups.entries())
    .map(([path, docs]) => {
      const best =
        docs.reduce(
          (a, b) => ((b.score ?? 0) > (a.score ?? 0) ? b : a),
          docs[0]
        ) || docs[0];

      return {
        title: extractTitle(best?.content || "", best?.file?.name, path),
        url: toDocUrl(best?.meta?.url),
        snippet: buildSnippet(best?.content || "", query),
        path,
        score: best?.score,
        category: categorizeDocument(best, path),
        type: best?.meta?.type,
      };
    })
    .sort((a, b) => (b.score ?? 0) - (a.score ?? 0));
};

export default function SearchBar() {
  const [query, setQuery] = useState("");
  const [showModal, setShowModal] = useState(false);
  const [activeFilter, setActiveFilter] = useState("all");
  const [isSearching, setIsSearching] = useState(false);
  const [error, setError] = useState(null);
  const [results, setResults] = useState([]);

  const searchInputRef = useRef(null);
  const modalRef = useRef(null);
  const requestAbortRef = useRef(null);

  const history = useHistory();

  const performSearch = useCallback(
    async (searchQuery) => {
      if (requestAbortRef.current) {
        requestAbortRef.current.abort();
      }
      const controller = new AbortController();
      requestAbortRef.current = controller;

      setIsSearching(true);
      setError(null);

      try {
        const response = await fetch(`/api/search`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ query: searchQuery, filter: activeFilter }),
          signal: controller.signal,
        });

        if (!response.ok) {
          throw new Error(`API error: ${response.statusText}`);
        }

        const data = await response.json();
        if (controller.signal.aborted) return;

        const documents = data?.results?.[0]?.documents || [];
        setResults(toResults(documents, searchQuery));
      } catch (err) {
        if (err?.name === "AbortError") return;
        setError("Failed to fetch search results.");
        setResults([]);
      } finally {
        if (!requestAbortRef.current?.signal.aborted) {
          setIsSearching(false);
          // setShowResults(true);
        }
      }
    },
    [activeFilter]
  );

  const debouncedSearch = useMemo(
    () => debounce(performSearch, DEBOUNCE_DELAY),
    [performSearch]
  );

  useEffect(() => {
    return () => {
      debouncedSearch.cancel();
      if (requestAbortRef.current) {
        requestAbortRef.current.abort();
      }
    };
  }, [debouncedSearch]);

  // Filter results based on active filter
  const filteredResults = useMemo(() => {
    if (activeFilter === "all") return results;
    return results.filter((result) => result.category === activeFilter);
  }, [results, activeFilter]);

  const handleInputChange = useCallback(
    (e) => {
      const value = e.target.value;
      setQuery(value);

      if (value.length < MIN_QUERY_LENGTH) {
        debouncedSearch.cancel();
        if (requestAbortRef.current) requestAbortRef.current.abort();
        setResults([]);
        setIsSearching(false);
        setError(null);
        return;
      }

      debouncedSearch(value);
    },
    [debouncedSearch]
  );

  const handleResultClick = useCallback(
    (url) => {
      history.push(url);
      setShowModal(false);
      setQuery("");
      setResults([]);
    },
    [history]
  );

  // Close modal when clicking outside or pressing Escape
  useEffect(() => {
    function handleClickOutside(event) {
      if (
        modalRef.current &&
        event.target instanceof Node &&
        !modalRef.current.contains(event.target)
      ) {
        setShowModal(false);
      }
    }

    function handleEscape(event) {
      if (event.key === "Escape") {
        setShowModal(false);
      }
    }

    if (showModal) {
      document.addEventListener("mousedown", handleClickOutside);
      document.addEventListener("keydown", handleEscape);
      document.body.style.overflow = "hidden"; // Prevent background scrolling
    }

    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
      document.removeEventListener("keydown", handleEscape);
      document.body.style.overflow = "unset";
    };
  }, [showModal]);

  const handleKeyDown = useCallback(
    (e) => {
      if (e.key === "Enter" && query.trim().length >= MIN_QUERY_LENGTH) {
        debouncedSearch.cancel();
        performSearch(query.trim());
      }
    },

    [debouncedSearch, performSearch, query]
  );

  // TODO: Update this after having the Haystack API endpoint
  const filters = [
    { id: "all", label: "All" },
    { id: "api-reference", label: "API Reference" },
    { id: "documentation", label: "Documentation" },
  ];

  return (
    <div style={{ display: "flex", gap: "0.5rem", alignItems: "center" }}>
      <div className={styles.searchContainer} role="search">
        <button
          className={styles.searchButton}
          onClick={() => setShowModal(true)}
          aria-label="Open search"
        >
          <span className={styles.searchIcon}>🔍</span>
          <span className={styles.searchPlaceholder}>
            Search documentation...
          </span>
        </button>
      </div>

      {showModal && (
        <div className={styles.modalOverlay}>
          <div className={styles.modalContent} ref={modalRef}>
            <div className={styles.modalHeader}>
              <div className={styles.searchInputWrapper} ref={searchInputRef}>
                <span className={styles.searchIconInput}>🔍</span>
                <input
                  type="text"
                  className={styles.searchInput}
                  placeholder="Search documentation..."
                  aria-label="Search documentation"
                  value={query}
                  onChange={handleInputChange}
                  onKeyDown={handleKeyDown}
                  autoFocus
                />
                {query && (
                  <button
                    type="button"
                    className={styles.clearButton}
                    aria-label="Clear search"
                    onClick={() => {
                      setQuery("");
                      setResults([]);
                      setError(null);
                      debouncedSearch.cancel();
                      if (requestAbortRef.current)
                        requestAbortRef.current.abort();
                    }}
                  >
                    ✕
                  </button>
                )}
                {isSearching && (
                  <div
                    className={styles.searchingIndicator}
                    role="status"
                    aria-live="polite"
                    aria-label="Searching"
                  >
                    <span className={styles.spinner} aria-hidden="true"></span>
                  </div>
                )}
              </div>

              <div className={styles.filterTabs} role="tablist">
                {filters.map((filter) => (
                  <button
                    key={filter.id}
                    className={`${styles.filterTab} ${
                      activeFilter === filter.id ? styles.filterTabActive : ""
                    }`}
                    onClick={() => {
                      setActiveFilter(filter.id);
                    }}
                    role="tab"
                    aria-selected={activeFilter === filter.id}
                  >
                    {filter.label}
                  </button>
                ))}
              </div>
            </div>

            <div className={styles.modalBody}>
              {!query || query.length < MIN_QUERY_LENGTH ? (
                <div className={styles.emptyState}>
                  <span className={styles.emptyStateIcon}>🔍</span>
                  <p className={styles.emptyStateText}>
                    Start typing to search...
                  </p>
                </div>
              ) : isSearching ? (
                <div className={styles.loadingState}>
                  <span className={styles.spinner}></span>
                  <p>Searching...</p>
                </div>
              ) : filteredResults.length > 0 ? (
                <div className={styles.searchResults}>
                  <ul>
                    {filteredResults.map((result, index) => (
                      <li
                        key={index}
                        onClick={() => handleResultClick(result.url)}
                      >
                        <div className={styles.resultTitle}>
                          <a
                            href={result.url}
                            onClick={(e) => {
                              e.preventDefault();
                              handleResultClick(result.url);
                            }}
                          >
                            {result.title}
                          </a>
                        </div>
                        <div
                          className={styles.resultSnippet}
                          dangerouslySetInnerHTML={{ __html: result.snippet }}
                        />
                      </li>
                    ))}
                  </ul>
                </div>
              ) : (
                <div className={styles.noResultsState}>
                  {error ? (
                    <p className={styles.noResults}>{error}</p>
                  ) : (
                    <p className={styles.noResults}>
                      No results found for "{query}"
                      {activeFilter !== "all" && ` in ${activeFilter}`}
                    </p>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
