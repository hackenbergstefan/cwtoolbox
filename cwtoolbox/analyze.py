# SPDX-FileCopyrightText: 2023 Stefan Hackenberg
#
# SPDX-License-Identifier: MIT

import random

import lascar
import numpy as np
from numba import njit

lascar.logger.setLevel(lascar.logging.CRITICAL)
np.seterr(divide="ignore")


@njit()
def u32_be(values, offset=0):
    """
    Convert list of bytes to u32 big endian.
    Similar to `int.from_bytes(values, "big")` but in nopython mode
    """
    return (
        values[4 * offset + 3]
        + (values[4 * offset + 2] << 8)
        + (values[4 * offset + 1] << 16)
        + (values[4 * offset + 0] << 24)
    )


@njit()
def u32_le(values, offset=0):
    """
    Convert list of bytes to u32 little endian.
    Similar to `int.from_bytes(values, "little")` but in nopython mode
    """
    return (
        values[4 * offset + 0]
        + (values[4 * offset + 1] << 8)
        + (values[4 * offset + 2] << 16)
        + (values[4 * offset + 3] << 24)
    )


def cpa(dataset, selection_functions, guess_range=range(1), higherorder_rois=None):
    class CpaOutput(lascar.OutputMethod):
        def __init__(self, *engines):
            super().__init__(*engines)
            self.result = []
            self.name = engines[0].name

        def _update(self, engine, results):
            if len(results) > 1:
                # pylint: disable=protected-access
                self.result = list(zip(engine._guess_range, results))
            else:
                self.result = results[0]

    trace = lascar.TraceBatchContainer(dataset["trace"], dataset)
    if higherorder_rois:
        trace.leakage_processing = lascar.CenteredProductProcessing(
            container=trace,
            rois=higherorder_rois,
            batch_size=100_000,
        )

    selection_functions = {name: njit()(f) for name, f in selection_functions.items()}
    engines = [
        lascar.CpaEngine(
            name=name,
            selection_function=f,
            guess_range=guess_range,
            jit=False,
        )
        for name, f in selection_functions.items()
    ]
    output_methods = [CpaOutput(engine) for engine in engines]
    session = lascar.Session(
        trace,
        engines=engines,
        output_method=output_methods,
        progressbar=False,
    )
    session.run(batch_size=100_000)
    return [
        (output_method.name, output_method.result) for output_method in output_methods
    ]


def cpa_ranking(
    dataset,
    selection_function,
    correct_key,
    guess_range=range(256),
    higherorder_rois=None,
):
    class CpaOutput(lascar.OutputMethod):
        def __init__(self, *engines):
            super().__init__(*engines)
            self.result = []
            self.name = engines[0].name

        def _update(self, engine, results):
            maxs = np.argsort(np.max(np.abs(results), axis=1))[::-1]
            self.result.append(
                (engine.finalize_step[-1], np.where(maxs == correct_key)[0][0])
            )

    trace = lascar.TraceBatchContainer(dataset["trace"], dataset)
    if higherorder_rois:
        trace.leakage_processing = lascar.CenteredProductProcessing(
            container=trace,
            rois=higherorder_rois,
            batch_size=100_000,
        )

    engine = lascar.CpaEngine(
        name="cpa",
        selection_function=selection_function,
        guess_range=guess_range,
    )
    output_method = CpaOutput(engine)
    session = lascar.Session(
        trace,
        engine=engine,
        output_method=output_method,
        output_steps=range(0, len(dataset), 100),
        progressbar=False,
    )
    session.run(batch_size=100_000)
    return output_method.result


def cpa_leakage_rate(dataset, selection_function, randoms=16):
    class CpaOutput(lascar.OutputMethod):
        def __init__(self, *engines):
            super().__init__(*engines)
            self.result = []

        def _update(self, engine, results):
            resabs = np.abs(results)
            self.result.append(
                (engine.finalize_step[-1], -np.std(results[1:]) / np.max(resabs[0]))
            )

    trace = lascar.TraceBatchContainer(dataset["trace"], dataset)

    selection_function = njit()(selection_function)
    engine = lascar.CpaEngine(
        name="cpa",
        selection_function=lambda value, guess: selection_function(value)
        if guess == 0
        else random.randint(0, 1),
        guess_range=range(randoms),
    )
    output_method = CpaOutput(engine)
    session = lascar.Session(
        trace,
        engine=engine,
        output_method=output_method,
        output_steps=range(0, len(dataset), 100),
        progressbar=False,
    )
    session.run(batch_size=100_000)
    return output_method.result


def ttest(dataset, selection_function, correct_key):
    class TtestOutput(lascar.OutputMethod):
        def __init__(self, *engines):
            super().__init__(*engines)
            self.result = []

        def _update(self, engine, results):
            guess = int(engine.name.split(" ")[-1])
            self.result.append((guess, results))

    trace = lascar.TraceBatchContainer(dataset["trace"], dataset["input"])

    engines = [
        lascar.TTestEngine(
            name=f"ttest {guess}",
            partition_function=selection_function(guess),
        )
        for guess in set([random.randint(0, 255) for _ in range(20)] + [correct_key])
    ]
    output_method = TtestOutput(*engines)
    session = lascar.Session(
        trace,
        engines=engines,
        output_method=output_method,
        progressbar=False,
    )
    session.run(batch_size=100_000)
    return output_method.result


def snr(dataset, selection_function, selection_range, correct_key):
    class SnrOutput(lascar.OutputMethod):
        def __init__(self, *engines):
            super().__init__(*engines)
            self.result = []

        def _update(self, engine, results):
            guess = int(engine.name.split(" ")[-1])
            self.result.append((guess, results))

    trace = lascar.TraceBatchContainer(dataset["trace"], dataset["input"])

    engines = [
        lascar.SnrEngine(
            name=f"ttest {guess}",
            partition_function=selection_function(guess),
            partition_range=selection_range,
        )
        for guess in set([random.randint(0, 255) for _ in range(20)] + [correct_key])
    ]
    output_method = SnrOutput(*engines)
    session = lascar.Session(
        trace,
        engines=engines,
        output_method=output_method,
        progressbar=False,
    )
    session.run(batch_size=100_000)
    return output_method.result


def cpa_evolution(dataset, selection_function, guess_range=range(256)):
    class CpaOutput(lascar.OutputMethod):
        def __init__(self, *engines):
            super().__init__(*engines)
            self.result = {guess: [] for guess in guess_range}

        def _update(self, engine, results):
            maxs = np.max(np.abs(results), axis=1)
            for guess, result in zip(guess_range, maxs):
                self.result[guess].append((engine.finalize_step[-1], result))

    trace = lascar.TraceBatchContainer(dataset["trace"], dataset)

    engine = lascar.CpaEngine(
        name="cpa",
        selection_function=selection_function,
        guess_range=guess_range,
    )
    output_method = CpaOutput(engine)
    session = lascar.Session(
        trace,
        engine=engine,
        output_method=output_method,
        output_steps=range(0, len(dataset), 100),
        progressbar=False,
    )
    session.run(batch_size=100_000)
    return list(output_method.result.items())
