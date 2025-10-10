// SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
//
// SPDX-License-Identifier: Apache-2.0

import {useEffect} from 'react';
import {useHistory} from '@docusaurus/router';

export default function Home() {
  const history = useHistory();

  useEffect(() => {
    // Redirect to the docs intro page immediately
    history.replace('/docs/overview/intro');
  }, [history]);

  // Return null since we're redirecting immediately
  return null;
}
