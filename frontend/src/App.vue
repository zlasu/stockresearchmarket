<template>
  <div class="app-shell">
    <aside class="rail">
      <RouterLink class="rail-logo" to="/" aria-label="Dashboard">
        <ActivityIcon :size="24" />
      </RouterLink>
      <RouterLink class="rail-button" to="/" title="Experiments">
        <ChartScatterIcon :size="20" />
      </RouterLink>
      <button class="rail-button" type="button" title="Refresh index" @click="refresh">
        <RefreshCwIcon :size="18" />
      </button>
    </aside>
    <main class="workspace">
      <RouterView />
    </main>
  </div>
</template>

<script setup lang="ts">
import { ActivityIcon, ChartScatterIcon, RefreshCwIcon } from "lucide-vue-next";

import { useResearchStore } from "./stores/research";

const store = useResearchStore();

async function refresh() {
  await store.refresh();
}
</script>

