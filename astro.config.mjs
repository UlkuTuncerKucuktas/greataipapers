// @ts-check
import mdx from '@astrojs/mdx';
import sitemap from '@astrojs/sitemap';
import { defineConfig } from 'astro/config';

export default defineConfig({
    site: 'https://greataipapers.com',
    integrations: [mdx(), sitemap()],
    image: {
        service: { entrypoint: 'astro/assets/services/noop' },
    },
});
