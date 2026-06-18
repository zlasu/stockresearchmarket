<template>
  <section v-if="detail" class="page">
    <header class="topbar">
      <div>
        <RouterLink class="back-link" to="/">Back to experiments</RouterLink>
        <p class="eyebrow">{{ detail.kind }}</p>
        <h1>{{ detail.name }}</h1>
      </div>
      <RouterLink class="button" :to="comparePath">Compare selected</RouterLink>
    </header>

    <div class="kpi-grid">
      <KpiCard label="Variants" :value="String(detail.variant_count)" />
      <KpiCard label="Best CAGR" :value="formatMetric('cagr', detail.metrics.cagr)" />
      <KpiCard label="Best Sharpe" :value="formatMetric('sharpe', detail.metrics.sharpe)" />
      <KpiCard label="Max DD" :value="formatMetric('drawdown_magnitude', detail.metrics.drawdown_magnitude)" />
    </div>

    <div v-if="detail.caveats.length" class="notice">
      <strong>Data caveats</strong>
      <span v-for="caveat in detail.caveats" :key="caveat">{{ caveat }}</span>
    </div>

    <section class="recommendation-grid">
      <article v-for="item in detail.recommendations" :key="item.title + item.variant_id" class="recommendation">
        <span>{{ item.title }}</span>
        <strong>{{ item.variant_name }}</strong>
        <p>{{ item.body }}</p>
      </article>
    </section>

    <div class="split-grid">
      <ChartCard v-if="scatterOption" title="Variant map" subtitle="Capital versus max drawdown magnitude." :option="scatterOption" />
      <section class="panel static-panel">
        <header class="panel-header">
          <div>
            <h2>Static diagnostics</h2>
            <p>Python-rendered chart using the Seaborn-style backend theme.</p>
          </div>
        </header>
        <img :src="api.plotUrl(detail.id, 'risk-return')" alt="Risk return plot" />
      </section>
    </div>

    <section class="panel">
      <header class="panel-header">
        <div>
          <h2>Variants</h2>
          <p>Select multiple rows, sort locally, then compare.</p>
        </div>
        <select v-model="variantSort" class="input">
          <option value="computed_score">Research score</option>
          <option value="cagr">CAGR</option>
          <option value="sharpe">Sharpe</option>
          <option value="drawdown_magnitude">Drawdown</option>
          <option value="avg_annual_turnover">Turnover</option>
        </select>
      </header>
      <div class="variant-list">
        <article v-for="variant in sortedVariants" :key="variant.id" class="variant-card">
          <label class="check-row">
            <input type="checkbox" :checked="selected.includes(variant.id)" @change="toggle(variant.id)" />
            <span>{{ variant.name }}</span>
          </label>
          <div class="variant-metrics">
            <span>CAGR {{ formatMetric("cagr", variant.metrics.cagr) }}</span>
            <span>Sharpe {{ formatMetric("sharpe", variant.metrics.sharpe) }}</span>
            <span>DD {{ formatMetric("drawdown_magnitude", variant.metrics.drawdown_magnitude) }}</span>
            <span>Turn {{ formatMetric("avg_annual_turnover", variant.metrics.avg_annual_turnover) }}</span>
          </div>
          <RouterLink class="inline-link" :to="`/experiments/${detail.id}/variants/${variant.id}`">Open variant</RouterLink>
        </article>
      </div>
    </section>
  </section>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from "vue";
import { useRoute } from "vue-router";

import { api } from "../api";
import ChartCard from "../components/ChartCard.vue";
import KpiCard from "../components/KpiCard.vue";
import { variantScatterOption } from "../charts";
import { formatMetric, metricNumber } from "../lib/format";
import { useResearchStore } from "../stores/research";

const route = useRoute();
const store = useResearchStore();
const id = computed(() => String(route.params.id));
const detail = computed(() => store.details[id.value]);
const selected = ref<string[]>([]);
const variantSort = ref("computed_score");
const scatterOption = computed(() => {
  const scatter = store.scatters[id.value];
  return scatter ? variantScatterOption(scatter) : null;
});

const sortedVariants = computed(() => {
  const variants = [...(detail.value?.variants ?? [])];
  return variants.sort((a, b) => {
    const av = variantSort.value === "computed_score" ? a.computed_score : metricNumber(a.metrics, variantSort.value);
    const bv = variantSort.value === "computed_score" ? b.computed_score : metricNumber(b.metrics, variantSort.value);
    return (Number(bv) || -Infinity) - (Number(av) || -Infinity);
  });
});

const comparePath = computed(() => `/experiments/${id.value}/compare?selected=${selected.value.join(",")}`);

function toggle(variantId: string) {
  selected.value = selected.value.includes(variantId)
    ? selected.value.filter((item) => item !== variantId)
    : [...selected.value, variantId].slice(-8);
}

onMounted(async () => {
  const loaded = await store.loadExperiment(id.value);
  selected.value = loaded.variants.slice(0, 3).map((variant) => variant.id);
  await store.loadScatter(id.value);
});
</script>

