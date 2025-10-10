// SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
//
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';
import { useHistory } from '@docusaurus/router';

export default function SearchBar() {
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const history = useHistory();

  const handleSearch = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;
    setLoading(true);
    try {
      const res = await fetch('http://localhost:8000/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query }),
      });
      const data = await res.json();
      const topResult = data?.documents?.[0];
      if (topResult?.meta?.url) {
        history.push(topResult.meta.url);
      } else {
        alert('No match found.');
      }
    } catch (err) {
      console.error('Search failed:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <form onSubmit={handleSearch} style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
      <input
        type="search"
        placeholder="Search the docs..."
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        style={{ padding: '0.5rem', border: '1px solid #ccc', borderRadius: '4px', flex: '1' }}
      />
      <button
        type="submit"
        disabled={loading}
        style={{ padding: '0.5rem 1rem', backgroundColor: '#319795', color: 'white', border: 'none', borderRadius: '4px', cursor: 'pointer' }}
      >
        {loading ? '...' : 'Search'}
      </button>
    </form>
  );
}
