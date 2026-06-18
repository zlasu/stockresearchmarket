<template>
  <section class="page">
    <header class="topbar">
      <div>
        <p class="eyebrow">Local research lab</p>
        <h1>Experiment Dashboard</h1>
      </div>
      <div class="toolbar">
        <input v-model="query" class="input" placeholder="Search experiments..." @keyup.enter="load" />
        <select v-model="kind" class="input" @change="load">
          <option value="">All kinds</option>
          <option v-for="item in kinds" :key="item" :value="item">{{ item }}</option>
        </select>
        <select v-model="sort" class="input" @change="load">
          <option value="created_at">Latest</option>
          <option value="cagr">CAGR</option>
          <option value="sharpe">Sharpe</option>
          <option value="max_drawdown">Drawdown</option>
          <option value="variant_count">Variants</option>
        </select>
      </div>
    </header>

    <div class="kpi-grid">
      <KpiCard label="Experiments" :value="String(store.total)" hint="indexed from experiments/" />
      <KpiCard label="Visible variants" :value="String(variantCount)" />
      <KpiCard label="Best CAGR" :value="formatMetric('cagr', bestCagr)" />
      <KpiCard label="Best Sharpe" :value="formatMetric('sharpe', bestSharpe)" />
    </div>

    <div v-if="store.error" class="notice danger">{{ store.error }}</div>

    <ChartCard
      v-if="store.experiments.length"
      title="Risk and return map"
      subtitle="Each point is one experiment, sized by variant count."
      :option="scatterOption"
    />

    <section class="panel">
      <header class="panel-header">
        <div>
          <h2>Experiments</h2>
          <p>Read-only index of local artifacts.</p>
        </div>
        <button class="button" type="button" @click="refresh">Refresh</button>
      </header>
      <div class="experiment-list">
        <RouterLink v-for="experiment in store.experiments" :key="experiment.id" class="experiment-row" :to="`/experiments/${experiment.id}`">
          <div>
            <strong>{{ experiment.name }}</strong>
            <span>{{ experiment.kind }} / {{ experiment.path }}</span>
          </div>
          <div class="metric-pill">{{ experiment.variant_count }} variants</div>
          <div class="metric-pill">CAGR {{ formatMetric("cagr", experiment.metrics.cagr) }}</div>
          <div class="metric-pill">DD {{ formatMetric("drawdown_magnitude", experiment.metrics.drawdown_magnitude) }}</div>
        </RouterLink>
      </div>
    </section>
  </section>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from "vue";

import ChartCard from "../components/ChartCard.vue";
import KpiCard from "../components/KpiCard.vue";
import { experimentScatterOption } from "../charts";
import { formatMetric, metricNumber } from "../lib/format";
import { useResearchStore } from "../stores/research";

const store = useResearchStore();
const query = ref("");
const kind = ref("");
const sort = ref("created_at");

const kinds = computed(() => [...new Set(store.experiments.map((item) => item.kind))].sort());
const variantCount = computed(() => store.experiments.reduce((sum, item) => sum + item.variant_count, 0));
const bestCagr = computed(() => Math.max(...store.experiments.map((item) => metricNumber(item.metrics, "cagr") ?? -Infinity)));
const bestSharpe = computed(() => Math.max(...store.experiments.map((item) => metricNumber(item.metrics, "sharpe") ?? -Infinity)));
const scatterOption = computed(() => experimentScatterOption(store.experiments));

async function load() {
  await store.loadExperiments({ query: query.value, kind: kind.value, sort: sort.value, limit: 100 });
}

async function refresh() {
  await store.refresh();
}

onMounted(load);
</script>

