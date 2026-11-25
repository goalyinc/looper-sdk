# goaly-looper

The Python SDK of Looper, Goaly's hosted post-training service. Your program
holds the data, the rewards and the loop, and calls a few primitives —
`forward_backward`, `optim_step`, `sample`, `save_state` — that Looper runs on
GPUs it manages. Your machine needs none.

The documentation is at <https://looper.goaly.ai/wiki/>: a
[quickstart](https://looper.goaly.ai/wiki/quickstart.html), the models,
sessions, training, sampling and checkpoints, the API reference, and
tutorials.

This is a beta: the API may change between releases, so pin the version.

## Install

Python 3.12 or newer.

```bash
pip install "goaly-looper==0.1.0b5"
```

The import package is `looper_sdk`. The SDK sends token ids, never text, so
bring a tokenizer, such as `transformers`'.

## Authenticate

Every call needs a Looper API key. Create one in the portal,
<https://looper.goaly.ai>, under **API Keys** in the left navigation, and
put it in `LOOPER_API_KEY`, where the SDK reads it when you build a client:

```bash
export LOOPER_API_KEY=sk-...
```

Or pass it as `api_key=` in `ServiceClient.Config`, for example from your
secret store. Don't commit a key, or share a notebook that holds one.
Without a valid key, `build()` raises `Unauthenticated`.

Your organization also needs credit: without it, the service refuses new
work with `SpendRefused`.

[Session](https://looper.goaly.ai/wiki/session.html#authenticate) has the
details.
