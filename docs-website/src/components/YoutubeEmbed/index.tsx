// SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
//
// SPDX-License-Identifier: Apache-2.0

import React from "react";
import styles from "./styles.module.css";

interface YouTubeEmbedProps {
  videoId: string;
  title?: string;
  className?: string;
}

export default function YouTubeEmbed({
  videoId,
  title = "YouTube video",
  className,
}: YouTubeEmbedProps) {
  return (
    <div className={`${styles.wrapper} ${className || ""}`}>
      <div className={styles.aspectRatio}>
        <iframe
          src={`https://www.youtube.com/embed/${videoId}`}
          title={title}
          frameBorder="0"
          allow="autoplay; encrypted-media; picture-in-picture"
          allowFullScreen
          loading="lazy"
          className={styles.iframe}
        />
      </div>
    </div>
  );
}
